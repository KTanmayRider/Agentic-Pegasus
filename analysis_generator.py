import os
import io
import json
import threading
import concurrent.futures
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------------
# Step 0 (Why): Establish constants, folders, and small utilities to match Arch.md
# - We need output/data directory and JSON contracts paths used by downstream stages
# - We include helpers for safe analysis_id and directory creation
# ---------------------------------------------------------------------------------
# Step 0 (Done): Constants/utilities defined to ensure contract and filesystem safety

OUTPUT_DIR = os.path.join("output")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
DRILLDOWN_DIR = os.path.join(OUTPUT_DIR, "graphs", "drilldown")  # used later by other stages
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "graphs", "summary")      # used later by other stages

MASTER_CSV_PATH = 'GRT Operations (operations.grt) (8).csv'
OUTPUT_QUANTITATIVE_JSON = os.path.join(DATA_DIR, 'quantitative_results.json')
OUTPUT_QUALITATIVE_JSON = os.path.join(DATA_DIR, 'qualitative_analysis.json')

DIMENSIONS_TO_ANALYZE = ['Relevance', 'Correctness', 'Completeness']
MAX_WORKERS = 2

# Configure client focus and model mappings
CLIENT_MODEL_NAME = 'Gemini 2.5 pro'

# Step 0.1 (Why): Align model identifiers with the actual CSV headers so schema validation and downstream logic work.
#                 The provided CSV contains "Chatgpt", "Gemini", and "Claude" model columns.
# Step 0.1 (Done): Updated mappings to match CSV; removed "Ollama" and "Grok".
MODEL_MAPPING = {
    'Chatgpt': 'Chatgpt',
    'Gemini': 'Gemini',
    'Claude': 'Claude'
}

# Step 0.2 (Why): Provide human-friendly display names for current models used in narratives and prompts.
# Step 0.2 (Done): Display names updated; client model remains Gemini 2.5 pro.
MODEL_DISPLAY_NAMES = {
    'Chatgpt': 'OpenAI o4-mini-high',
    'Gemini': 'Gemini 2.5 pro',
    'Claude': 'Claude Opus 4'
}

MODEL_NAMES = list(MODEL_MAPPING.keys())

COLUMN_NAMES = {
    'language': 'Code Language',
    'domain': 'Industry',
    'prompt_id': 'Prompt ID',
    'prompt': 'Prompt'
}

log_lock = threading.Lock()


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DRILLDOWN_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def sanitize_component(text: str) -> str:
    return ''.join(ch for ch in str(text) if ch.isalnum() or ch in ('-', '_')).replace(' ', '')


def make_analysis_id(language: str, domain: str, dimension: str) -> str:
    return f"{sanitize_component(language)}-{sanitize_component(domain)}-{sanitize_component(dimension)}"


def get_justification_column_name(model_name: str, dimension: str) -> str:
    prefix = MODEL_MAPPING[model_name]
    return f"{prefix} Human {dimension} Justification"


# ---------------------------------------------------------------------------------
# Step 1 (Why): Define state object and configuration to mirror original system
# - Add winner_text/client_performance_text to support qualitative JSON contract
# ---------------------------------------------------------------------------------
# Step 1 (Done): AnalysisState includes JSON-contract fields

@dataclass
class AnalysisState:
    language: str
    domain: str
    dimension: str
    data_slice: pd.DataFrame
    quantitative_context: Dict[str, Any] = field(default_factory=dict)
    is_data_valid: bool = False
    eda_report: str = ""
    analysis_report: str = ""
    error_message: str = ""
    validation_warnings: List[str] = field(default_factory=list)
    missing_justifications: Dict[str, List[int]] = field(default_factory=dict)
    winner_text: str = ""
    client_performance_text: str = ""

    def __post_init__(self):
        if not isinstance(self.data_slice, pd.DataFrame):
            self.data_slice = pd.DataFrame()


# ---------------------------------------------------------------------------------
# Step 2 (Why): Environment setup for Gemini API used by AnalysisAgent
# ---------------------------------------------------------------------------------
# Step 2 (Done): API key configured with safety checks

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise KeyError("GOOGLE_API_KEY not found in .env file.")
genai.configure(api_key=api_key)


# ---------------------------------------------------------------------------------
# Step 3 (Why): Configuration agent to validate CSV and set worker count
# ---------------------------------------------------------------------------------
# Step 3 (Done): ConfigurationAgent mirrors original behavior, minimal changes

class ConfigurationAgent:
    def __init__(self):
        self.validated_schema = False
        self.optimal_workers = MAX_WORKERS
        # Step 3.1 (Why): Track which models/dimensions are actually present to allow partial datasets.
        # Step 3.1 (Done): Initialize containers for availability reporting.
        self.available_models: List[str] = []
        self.available_dimensions_by_model: Dict[str, List[str]] = {}

    def validate_master_csv_schema(self, file_path: str) -> bool:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            missing_base_cols = [col for col in COLUMN_NAMES.values() if col not in df.columns]
            if missing_base_cols:
                print(f"❌ Missing base columns: {missing_base_cols}")
                return False

            # Step 3.1 (Why): Relax model-column validation so we continue even if some models/dimensions are missing.
            #                 This allows running when only Python rows or a subset of models are present.
            # Step 3.1 (Done): Log warnings for missing columns; require at least one score column overall.
            missing_cols: List[str] = []
            present_models: set[str] = set()
            dims_by_model: Dict[str, List[str]] = {m: [] for m in MODEL_NAMES}
            any_score_present = False

            for model_name in MODEL_MAPPING.keys():
                for dimension in DIMENSIONS_TO_ANALYZE:
                    score_col = f"{model_name} Human {dimension}"
                    just_col = get_justification_column_name(model_name, dimension)
                    if score_col in df.columns:
                        any_score_present = True
                        present_models.add(model_name)
                        dims_by_model[model_name].append(dimension)
                    else:
                        missing_cols.append(score_col)
                    if just_col not in df.columns:
                        missing_cols.append(just_col)

            if not any_score_present:
                print("❌ No model score columns detected. Ensure at least one '<Model> Human <Dimension>' column exists.")
                return False

            if missing_cols:
                print(f"⚠️ Some model columns are missing (processing will continue with available data): {missing_cols}")

            self.available_models = sorted(present_models)
            self.available_dimensions_by_model = {m: dims_by_model.get(m, []) for m in MODEL_NAMES}

            self.validated_schema = True
            print(
                f"Schema validation passed. Rows={len(df)}, Cols={len(df.columns)} | "
                f"Available models={self.available_models}"
            )
            return True
        except FileNotFoundError:
            print(f"❌ Master CSV not found: {file_path}")
            return False
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False

    def determine_optimal_workers(self) -> int:
        cpu_cores = os.cpu_count() or MAX_WORKERS
        self.optimal_workers = min(cpu_cores // 2, MAX_WORKERS)
        return self.optimal_workers


# ---------------------------------------------------------------------------------
# Step 4 (Why): DataSlicerTool computing quantitative analysis for the whole dataset
# - Provide aggregate structure that downstream graph_generator can consume
# ---------------------------------------------------------------------------------
# Step 4 (Done): Quantitative results kept in memory and exposed for aggregation

class DataSlicerTool:
    def __init__(self):
        self.quantitative_results: Dict[str, Dict[str, Any]] = {}
        self.quantitative_winners: Dict[str, Dict[str, Any]] = {}

    def compute_quantitative_analysis(self, df: pd.DataFrame) -> None:
        results: Dict[str, Dict[str, Any]] = {}

        for (language, domain), group_df in df.groupby([COLUMN_NAMES['language'], COLUMN_NAMES['domain']]):
            if pd.isna(language) or pd.isna(domain):
                continue
            key = f"{domain} ({language})"
            results[key] = {}
            for dimension in DIMENSIONS_TO_ANALYZE:
                results[key][dimension] = {}
                for model_name in MODEL_NAMES:
                    score_col = f"{model_name} Human {dimension}"
                    if score_col in group_df.columns:
                        scores = group_df[score_col].dropna()
                        score_counts = Counter(scores)
                        results[key][dimension][model_name] = {
                            'total_scores': int(len(scores)),
                            'score_5': int(score_counts.get(5, 0)),
                            'score_4': int(score_counts.get(4, 0)),
                            'score_3': int(score_counts.get(3, 0)),
                            'score_2': int(score_counts.get(2, 0)),
                            'score_1': int(score_counts.get(1, 0)),
                            'distribution': {int(k): int(v) for k, v in score_counts.items()}
                        }
        winners = self._determine_winners(results)
        self.quantitative_results = results
        self.quantitative_winners = winners

    def _determine_winners(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        winners: Dict[str, Dict[str, Any]] = {}
        for domain_lang, dimensions in results.items():
            winners[domain_lang] = {}
            for dimension, models in dimensions.items():
                model_scores = []
                for model_name, scores in models.items():
                    model_scores.append((
                        model_name,
                        scores['score_5'],
                        scores['score_4'],
                        scores['score_3'],
                        scores['total_scores']
                    ))
                # Sort for ranking display (5s, then 4s, then 3s)
                model_scores.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

                if not model_scores:
                    continue

                # Step (Tie Logic): Determine if there is a tie on top 5s
                top_fives = model_scores[0][1]
                tied = [m for m in model_scores if m[1] == top_fives]

                if len(tied) == 1:
                    top = tied[0]
                    winners[domain_lang][dimension] = {
                        'is_tie': False,
                        'winner': top[0],
                        'winner_5s': top[1],
                        'winner_4s': top[2],
                        'winner_3s': top[3],
                        'ranking': [m[0] for m in model_scores]
                    }
                else:
                    # Multi-model tie by number of 5s
                    winners[domain_lang][dimension] = {
                        'is_tie': True,
                        'winners': [
                            {
                                'model': m[0],
                                'score_5': m[1],
                                'score_4': m[2],
                                'score_3': m[3],
                                'total_scores': m[4]
                            } for m in tied
                        ],
                        'top_5s': top_fives,
                        'ranking': [m[0] for m in model_scores]
                    }
        return winners

    def create_analysis_states(self, master_df: pd.DataFrame) -> List['AnalysisState']:
        self.compute_quantitative_analysis(master_df)

        states: List[AnalysisState] = []
        base_columns = [col for col in COLUMN_NAMES.values() if col in master_df.columns]
        for (language, domain), group_df in master_df.groupby([COLUMN_NAMES['language'], COLUMN_NAMES['domain']]):
            if pd.isna(language) or pd.isna(domain):
                continue
            for dimension in DIMENSIONS_TO_ANALYZE:
                required_columns = base_columns.copy()
                for model_name in MODEL_MAPPING.keys():
                    score_col = f"{model_name} Human {dimension}"
                    just_col = get_justification_column_name(model_name, dimension)
                    if score_col in master_df.columns:
                        required_columns.append(score_col)
                    if just_col in master_df.columns:
                        required_columns.append(just_col)
                filtered_df = group_df[required_columns].copy()

                # extract context for this specific slice
                domain_key = f"{domain} ({language})"
                context = {'results': {}, 'winners': {}}
                if domain_key in self.quantitative_results and dimension in self.quantitative_results[domain_key]:
                    context['results'] = {domain_key: {dimension: self.quantitative_results[domain_key][dimension]}}
                if domain_key in self.quantitative_winners and dimension in self.quantitative_winners[domain_key]:
                    context['winners'] = {domain_key: {dimension: self.quantitative_winners[domain_key][dimension]}}

                states.append(AnalysisState(
                    language=language,
                    domain=domain,
                    dimension=dimension,
                    data_slice=filtered_df,
                    quantitative_context=context
                ))
        return states


# ---------------------------------------------------------------------------------
# Step 5 (Why): Validation agent (EDA) mirrors original with minimal changes
# ---------------------------------------------------------------------------------
# Step 5 (Done): EDAValidatorAgent implemented

class EDAValidatorAgent:
    def __init__(self):
        self.validation_rules = {
            'min_data_points': 1,
            'required_columns': ['Prompt ID'],
            'score_range': (1, 5)
        }

    def validate_data_slice(self, state: AnalysisState) -> AnalysisState:
        validation_warnings: List[str] = []
        missing_justifications: Dict[str, List[int]] = {}

        if state.data_slice.empty:
            state.is_data_valid = False
            state.eda_report = "❌ CRITICAL: No data found for this combination."
            state.error_message = "Empty data slice"
            return state

        if len(state.data_slice) < self.validation_rules['min_data_points']:
            validation_warnings.append(f"Low data points: {len(state.data_slice)} rows")

        missing_justifications = self._check_missing_justifications(state)
        score_issues = self._validate_scores(state)
        validation_warnings.extend(score_issues)

        eda_report = self._generate_eda_report(state, validation_warnings, missing_justifications)
        state.is_data_valid = True
        state.eda_report = eda_report
        state.validation_warnings = validation_warnings
        state.missing_justifications = missing_justifications
        return state

    def _check_missing_justifications(self, state: AnalysisState) -> Dict[str, List[int]]:
        missing: Dict[str, List[int]] = {}
        if 'Prompt ID' not in state.data_slice.columns:
            return missing
        for model_name in MODEL_NAMES:
            just_col = get_justification_column_name(model_name, state.dimension)
            if just_col in state.data_slice.columns:
                missing_prompts: List[int] = []
                for _, row in state.data_slice.iterrows():
                    prompt_id = row.get('Prompt ID')
                    justification = row.get(just_col)
                    if pd.isna(justification) or str(justification).strip() == '':
                        if prompt_id not in missing_prompts:
                            missing_prompts.append(prompt_id)
                if missing_prompts:
                    missing[model_name] = missing_prompts
        return missing

    def _validate_scores(self, state: AnalysisState) -> List[str]:
        issues: List[str] = []
        min_score, max_score = self.validation_rules['score_range']
        for model_name in MODEL_NAMES:
            score_col = f"{model_name} Human {state.dimension}"
            if score_col in state.data_slice.columns:
                scores = state.data_slice[score_col].dropna()
                invalid_scores = scores[(scores < min_score) | (scores > max_score)]
                if len(invalid_scores) > 0:
                    issues.append(f"{model_name}: {len(invalid_scores)} scores outside valid range")
                if len(scores) > 3 and scores.nunique() == 1:
                    issues.append(f"{model_name}: All scores identical ({scores.iloc[0]})")
        return issues

    def _generate_eda_report(self, state: AnalysisState, warnings: List[str], missing_just: Dict[str, List[int]]) -> str:
        lines: List[str] = []
        lines.append(f"Data Overview for {state.domain} ({state.language}) - {state.dimension}")
        lines.append(f"   - Total data points: {len(state.data_slice)}")
        if 'Prompt ID' in state.data_slice.columns:
            lines.append(f"   - Unique prompts: {state.data_slice['Prompt ID'].nunique()}")
        if missing_just:
            lines.append("\nMissing Justifications Detected:")
            for model_name, prompt_ids in missing_just.items():
                lines.append(f"   - {model_name}: {len(prompt_ids)} missing justifications (Prompts: {prompt_ids})")
        else:
            lines.append("\nJustifications Complete: All justifications present")
        lines.append("\nScore Distribution Summary:")
        for model_name in MODEL_NAMES:
            score_col = f"{model_name} Human {state.dimension}"
            if score_col in state.data_slice.columns:
                scores = state.data_slice[score_col].dropna()
                if len(scores) > 0:
                    score_counts = Counter(scores)
                    dist_str = ', '.join([f"{score}: {count}" for score, count in sorted(score_counts.items())])
                    lines.append(f"   - {model_name}: {dist_str}")
        if warnings:
            lines.append("\nValidation Warnings:")
            for w in warnings:
                lines.append(f"   - {w}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------------
# Step 6 (Why): Analysis agent uses Gemini to produce two texts per contract
# - We parse the response into winner_text and client_performance_text
# ---------------------------------------------------------------------------------
# Step 6 (Done): AnalysisAgent implemented with simple parsing rules

class AnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def generate_analysis(self, state: AnalysisState) -> AnalysisState:
        try:
            with io.StringIO() as buffer:
                state.data_slice.to_csv(buffer, index=False)
                csv_string = buffer.getvalue()
            prompt = self._create_enhanced_prompt(state, csv_string)
            response = self.model.generate_content(prompt)
            full_text = (response.text or "").strip()
            state.analysis_report = full_text
            # naive parsing based on required leading sentences
            client_header = f"{CLIENT_MODEL_NAME} has the following performance"
            idx = full_text.find(client_header)
            if idx != -1:
                state.winner_text = full_text[:idx].strip()
                state.client_performance_text = full_text[idx:].strip()
            else:
                state.winner_text = full_text
                state.client_performance_text = ""
            return state
        except Exception as e:
            state.error_message = f"Error generating analysis: {e}"
            state.winner_text = ""
            state.client_performance_text = ""
            state.analysis_report = state.error_message
            return state

    def _create_enhanced_prompt(self, state: AnalysisState, csv_data: str) -> str:
        domain_key = f"{state.domain} ({state.language})"
        score_summary = "**Score Summary for Context:**\n"
        winner_info = "**Winner**: Could not determine from available data.\n"
        if 'results' in state.quantitative_context and domain_key in state.quantitative_context['results']:
            if state.dimension in state.quantitative_context['results'][domain_key]:
                score_data = state.quantitative_context['results'][domain_key][state.dimension]
                for model_name in MODEL_NAMES:
                    if model_name in score_data:
                        scores = score_data[model_name]
                        display_name = MODEL_DISPLAY_NAMES[model_name]
                        score_summary += (
                            f"- {display_name}: {scores['score_5']} fives, {scores['score_4']} fours, "
                            f"{scores['score_3']} threes, {scores['score_2']} twos, {scores['score_1']} ones\n"
                        )
        if 'winners' in state.quantitative_context and domain_key in state.quantitative_context['winners']:
            if state.dimension in state.quantitative_context['winners'][domain_key]:
                w = state.quantitative_context['winners'][domain_key][state.dimension]
                if w['is_tie']:
                    winner_info = "**Winner**: Multiple models tied for the top performance.\n"
                    winner_info += "   - Top 5s: " + str(w['top_5s']) + "\n"
                    winner_info += "   - Winners:\n"
                    for winner_info_item in w['winners']:
                        winner_display_name = MODEL_DISPLAY_NAMES[winner_info_item['model']]
                        winner_info += f"     - {winner_display_name} (5s: {winner_info_item['score_5']})\n"
                else:
                    winner_display_name = MODEL_DISPLAY_NAMES[w['winner']]
                    winner_info = f"**Winner**: {winner_display_name} ({w['winner_5s']} fives)\n"

        prompt = f"""
You are an expert data analyst comparing LLM performance based on the provided CSV data.
The data covers the '{state.domain}' industry for the '{state.language}' language, focusing on the '{state.dimension}' dimension.

**INTERNAL EDA VALIDATION REPORT:**
{state.eda_report}

{score_summary}
{winner_info}

**CSV Data for Analysis:**
```csv
{csv_data}
```

**Your Task:**
Analyze the CSV data and provide a comprehensive analysis following these instructions in the exact order given:

1.  **Declare the Winner (or Tie):**
    - Determine the winner as the model with the most '5' ratings for '{state.dimension}'. If two or more models share the same maximum number of '5' ratings, declare a tie.
    - If there is a single winner, your response MUST start with: "The winner is [MODEL_NAME]."
    - If there is a tie, your response MUST start with: "The winners are [MODEL_NAME_1], [MODEL_NAME_2], ... (tie)."
    - Add a brief, positive analysis (2-3 sentences) explaining the result, including the exact number of 5s.
    - **IMPORTANT**: Include the specific number of 5s achieved by the winner(s) in your explanation.

2.  **Analyze {CLIENT_MODEL_NAME}'s Performance:**
    - This section MUST start with the exact sentence: "{CLIENT_MODEL_NAME} has the following performance in the {state.domain} industry :"
    - For each unique 'Prompt ID', write a flowing paragraph (not bullet points) that follows this format:
      "Prompt ID [NUMBER] ([DESCRIPTIVE TITLE]): [ANALYSIS]"
    - Create a descriptive title for each prompt based on the prompt content (e.g., "Simple 2D Physics Engine", "User Authentication System").
    - **CRITICAL**: In your analysis, ALWAYS mention the specific scores received (e.g., "received two 5's", "earned one perfect score", "achieved three 5 ratings").
    - Write in a narrative, analytical style similar to a performance review.
    - **Content Guidelines:**
        - If the ratings include a '5': Start by acknowledging the positive achievement with specific numbers ("While {CLIENT_MODEL_NAME} received [X] 5's for this prompt..."), then focus on negative feedback and areas for improvement.
        - If the ratings include **NO '5's**: Focus entirely on weaknesses and shortcomings. Do not mention any positives.
    - Use the justifications from the CSV as evidence, but synthesize them into flowing analysis rather than just quoting.
    - Each prompt analysis should be 3-4 sentences that provide context, performance assessment, and implications.
    - Use sophisticated analytical language and provide insights into what the performance means for real-world applications.

3. **Integration of EDA Findings:**
    - If the EDA report mentions missing justifications, acknowledge this in your analysis by noting where data gaps exist.
    - If there are validation warnings, briefly mention their potential impact on the analysis reliability.
    - Use the EDA insights to provide more nuanced interpretation of the results.

**Important Notes:**
- Pay close attention to the EDA validation report above, which has already analyzed the data for completeness and quality.
- If missing justifications are noted for specific prompts, mention this limitation in your analysis of those prompts.
- Maintain analytical rigor while acknowledging data quality issues identified by our validation process.
"""
        return prompt


# ---------------------------------------------------------------------------------
# Step 7 (Why): Error handler for resilience
# ---------------------------------------------------------------------------------
# Step 7 (Done): ErrorHandlerAgent implemented

class ErrorHandlerAgent:
    def handle_error(self, state: AnalysisState) -> AnalysisState:
        state.analysis_report = state.analysis_report or state.error_message or "Unknown error"
        return state


# ---------------------------------------------------------------------------------
# Step 8 (Why): Build a minimal LangGraph workflow: validate -> analysis -> end
# ---------------------------------------------------------------------------------
# Step 8 (Done): Workflow compiled and ready for thread execution

def create_analysis_workflow():
    eda = EDAValidatorAgent()
    analyst = AnalysisAgent()
    err = ErrorHandlerAgent()

    def eda_node(s: AnalysisState) -> AnalysisState:
        return eda.validate_data_slice(s)

    def analysis_node(s: AnalysisState) -> AnalysisState:
        return analyst.generate_analysis(s)

    def error_node(s: AnalysisState) -> AnalysisState:
        return err.handle_error(s)

    def route_after_validation(s: AnalysisState) -> str:
        if s.is_data_valid and not s.error_message:
            return "analysis"
        return "error"

    workflow = StateGraph(AnalysisState)
    workflow.add_node("eda", eda_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("error", error_node)
    workflow.set_entry_point("eda")
    workflow.add_conditional_edges(
        "eda",
        route_after_validation,
        {
            "analysis": "analysis",
            "error": "error"
        }
    )
    workflow.add_edge("analysis", END)
    workflow.add_edge("error", END)
    return workflow.compile()


# ---------------------------------------------------------------------------------
# Step 9 (Why): Aggregator to emit JSON contracts per Arch.md
# - quantitative_results.json: list of entries per analysis unit with scores + winners
# - qualitative_analysis.json: list of entries with analysis_id and required texts
# ---------------------------------------------------------------------------------
# Step 9 (Done): ReportAggregator writes both JSON files to output/data

class ReportAggregator:
    def __init__(self, slicer: DataSlicerTool):
        self.slicer = slicer
        self.quantitative_entries: List[Dict[str, Any]] = []
        self.qualitative_entries: List[Dict[str, Any]] = []

    def collect_quantitative(self):
        entries: List[Dict[str, Any]] = []
        for domain_lang, dims in self.slicer.quantitative_results.items():
            # domain_lang is "{domain} ({language})"
            try:
                domain, lang_part = domain_lang.split(' (')
                language = lang_part[:-1]
            except Exception:
                # fallback parse
                parts = domain_lang.split('|')
                domain = parts[0] if parts else domain_lang
                language = parts[1] if len(parts) > 1 else "Unknown"
            for dimension, models in dims.items():
                analysis_id = make_analysis_id(language, domain, dimension)
                winner_info = self.slicer.quantitative_winners.get(domain_lang, {}).get(dimension, {})
                entries.append({
                    'analysis_id': analysis_id,
                    'language': language,
                    'domain': domain,
                    'dimension': dimension,
                    'models': models,
                    'winner': winner_info
                })
        self.quantitative_entries = entries

    def collect_qualitative(self, completed_states: List[AnalysisState]):
        q_entries: List[Dict[str, Any]] = []
        for s in completed_states:
            analysis_id = make_analysis_id(s.language, s.domain, s.dimension)
            q_entries.append({
                'analysis_id': analysis_id,
                'language': s.language,
                'domain': s.domain,
                'dimension': s.dimension,
                'winner_text': s.winner_text,
                'client_performance_text': s.client_performance_text,
                'is_error': bool(s.error_message)
            })
        self.qualitative_entries = q_entries

    def save_json_contracts(self):
        ensure_dirs()
        with open(OUTPUT_QUANTITATIVE_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.quantitative_entries, f, indent=2, ensure_ascii=False)
        with open(OUTPUT_QUALITATIVE_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.qualitative_entries, f, indent=2, ensure_ascii=False)
        print(f"Wrote quantitative -> {OUTPUT_QUANTITATIVE_JSON}")
        print(f"Wrote qualitative -> {OUTPUT_QUALITATIVE_JSON}")


# ---------------------------------------------------------------------------------
# Step 10 (Why): Main orchestration for Stage 1 (analysis_generator)
# ---------------------------------------------------------------------------------
# Step 10 (Done): End-to-end run creating the two JSON contract files

def main():
    print("analysis_generator starting (Stage 1)")
    ensure_dirs()

    # Initialize config and validate schema
    cfg = ConfigurationAgent()
    if not cfg.validate_master_csv_schema(MASTER_CSV_PATH):
        print("❌ Schema validation failed. Exiting.")
        return
    cfg.determine_optimal_workers()

    # Load CSV
    try:
        master_df = pd.read_csv(MASTER_CSV_PATH, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to load master CSV: {e}")
        return

    # Create states
    slicer = DataSlicerTool()
    states = slicer.create_analysis_states(master_df)
    if not states:
        print("❌ No analysis states created. Exiting.")
        return

    # Compile workflow
    workflow = create_analysis_workflow()

    # Execute in parallel
    completed: List[AnalysisState] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.optimal_workers) as executor:
        futures = [executor.submit(lambda st: workflow.invoke(st), s) for s in states]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            # LangGraph may return dict-like. Normalize to AnalysisState
            if isinstance(res, dict):
                s = AnalysisState(
                    language=res.get('language', 'Unknown'),
                    domain=res.get('domain', 'Unknown'),
                    dimension=res.get('dimension', 'Unknown'),
                    data_slice=pd.DataFrame(),
                )
                s.eda_report = res.get('eda_report', '')
                s.analysis_report = res.get('analysis_report', '')
                s.error_message = res.get('error_message', '')
                s.winner_text = res.get('winner_text', '')
                s.client_performance_text = res.get('client_performance_text', '')
                completed.append(s)
            else:
                completed.append(res)

    # Aggregate and save contracts
    agg = ReportAggregator(slicer)
    agg.collect_quantitative()
    agg.collect_qualitative(completed)
    agg.save_json_contracts()
    print("🎉 analysis_generator complete")


if __name__ == "__main__":
    main() 