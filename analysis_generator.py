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
OUTPUT_OVERALL_QUAL_JSON = os.path.join(DATA_DIR, 'overall_qualitative_analysis.json')
# System-eval outputs (CodeBLEU and related metrics)
OUTPUT_SYSTEM_QUANT_JSON = os.path.join(DATA_DIR, 'system_quantitative_results.json')
OUTPUT_SYSTEM_QUAL_JSON = os.path.join(DATA_DIR, 'system_qualitative_analysis.json')
# System-eval CSV outputs for easy verification/graphing
OUTPUT_SYSTEM_QUANT_CSV = os.path.join(DATA_DIR, 'system_quantitative_results.csv')
OUTPUT_SYSTEM_QUAL_CSV = os.path.join(DATA_DIR, 'system_qualitative_analysis.csv')
# Sample-style RLHF system eval CSV (per-prompt rows)
OUTPUT_RLHF_SYSTEM_EVALS_CSV = os.path.join(DATA_DIR, 'RLHF data - System evals.csv')

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
    'prompt': 'Prompt',
    'topic': 'Topic'
}

# System evaluation metric columns (present in the master CSV)
SYSTEM_METRICS = [
    'Codebleu Score',
    'Ngram Score',
    'Weight Ngram Score',
    'Dataflow Match Score'
]

# Display names for system CSV header to match sample file
SYSTEM_CSV_DISPLAY_NAMES = {
    'Chatgpt': 'Chat GPT o4 mini-high',
    'Gemini': 'Gemini 2.5 pro',
    'Claude': 'Claude Opus 4'
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
    is_overall: bool = False

    def __post_init__(self):
        if not isinstance(self.data_slice, pd.DataFrame):
            self.data_slice = pd.DataFrame()


# System evaluation state for CodeBLEU and related metrics
@dataclass
class SystemAnalysisState:
    language: str
    domain: str
    data_slice: pd.DataFrame
    system_context: Dict[str, Any] = field(default_factory=dict)
    is_data_valid: bool = False
    eda_report: str = ""
    analysis_report: str = ""
    error_message: str = ""
    validation_warnings: List[str] = field(default_factory=list)
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
        # System metrics availability
        self.available_system_metrics_by_model: Dict[str, List[str]] = {}

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

            # Detect available system metrics per model (optional, does not affect pass/fail)
            sys_avail: Dict[str, List[str]] = {m: [] for m in MODEL_NAMES}
            for model_name in MODEL_MAPPING.keys():
                for metric in SYSTEM_METRICS:
                    if f"{model_name} {metric}" in df.columns:
                        sys_avail[model_name].append(metric)
            self.available_system_metrics_by_model = sys_avail

            self.validated_schema = True
            print(
                f"Schema validation passed. Rows={len(df)}, Cols={len(df.columns)} | "
                f"Available models={self.available_models} | System metrics={self.available_system_metrics_by_model}"
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
        # System-eval aggregates
        self.system_results: Dict[str, Dict[str, Any]] = {}
        self.system_winners: Dict[str, Dict[str, Any]] = {}

    def compute_quantitative_analysis(self, df: pd.DataFrame) -> None:
        results: Dict[str, Dict[str, Any]] = {}

        for (language, domain), group_df in df.groupby([COLUMN_NAMES['language'], COLUMN_NAMES['domain']]):
            if pd.isna(language) or pd.isna(domain):
                continue
            key = f"{domain} ({language})"
            results[key] = {}
            for dimension in DIMENSIONS_TO_ANALYZE:
                results[key][dimension] = {}
                # Per-model aggregate (backward-compatible flat layout)
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
                # Per-topic breakdown by model (for graphing)
                topics_map: Dict[str, Dict[str, Any]] = {}
                topic_col = COLUMN_NAMES.get('topic')
                if topic_col and topic_col in group_df.columns:
                    for topic_value, topic_df in group_df.groupby(topic_col):
                        # Skip NaN topic values
                        if pd.isna(topic_value):
                            continue
                        per_model: Dict[str, Any] = {}
                        for model_name in MODEL_NAMES:
                            score_col = f"{model_name} Human {dimension}"
                            if score_col in topic_df.columns:
                                t_scores = topic_df[score_col].dropna()
                                t_counts = Counter(t_scores)
                                per_model[model_name] = {
                                    'total_scores': int(len(t_scores)),
                                    'score_5': int(t_counts.get(5, 0)),
                                    'score_4': int(t_counts.get(4, 0)),
                                    'score_3': int(t_counts.get(3, 0)),
                                    'score_2': int(t_counts.get(2, 0)),
                                    'score_1': int(t_counts.get(1, 0)),
                                    'distribution': {int(k): int(v) for k, v in t_counts.items()}
                                }
                        if per_model:
                            topics_map[str(topic_value)] = per_model
                if topics_map:
                    results[key][dimension]['topics'] = topics_map
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
                    # Skip non-model buckets such as 'topics'
                    if model_name == 'topics' or not isinstance(scores, dict):
                        continue
                    if 'score_5' not in scores:
                        continue
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

    # System evaluation: compute aggregates across metrics per (language, domain)
    def compute_system_evaluation(self, df: pd.DataFrame) -> None:
        sys_results: Dict[str, Dict[str, Any]] = {}
        for (language, domain), group_df in df.groupby([COLUMN_NAMES['language'], COLUMN_NAMES['domain']]):
            if pd.isna(language) or pd.isna(domain):
                continue
            key = f"{domain} ({language})"
            metric_block: Dict[str, Any] = {}
            # Per-model metric aggregates (mean/std/count)
            for model_name in MODEL_NAMES:
                per_metric: Dict[str, Any] = {}
                for metric in SYSTEM_METRICS:
                    col = f"{model_name} {metric}"
                    if col in group_df.columns:
                        series = pd.to_numeric(group_df[col], errors='coerce').dropna()
                        if len(series) > 0:
                            per_metric[metric] = {
                                'mean': float(series.mean()),
                                'std': float(series.std(ddof=0)) if len(series) > 1 else 0.0,
                                'count': int(series.shape[0])
                            }
                if per_metric:
                    metric_block[model_name] = per_metric
            if metric_block:
                sys_results[key] = {'metrics': metric_block}
        self.system_results = sys_results
        self.system_winners = self._determine_system_winners(sys_results)

    def _determine_system_winners(self, sys_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        winners: Dict[str, Dict[str, Any]] = {}
        for domain_lang, payload in sys_results.items():
            metric_block = payload.get('metrics', {})
            # Determine winner by highest mean CodeBLEU
            table = []
            for model_name, metrics in metric_block.items():
                codebleu = metrics.get('Codebleu Score', {}).get('mean', None)
                if codebleu is not None:
                    table.append((model_name, codebleu))
            if not table:
                continue
            table.sort(key=lambda x: x[1], reverse=True)
            top_score = table[0][1]
            tied = [m for m in table if m[1] == top_score]
            if len(tied) == 1:
                winners[domain_lang] = {
                    'is_tie': False,
                    'winner': tied[0][0],
                    'winner_codebleu': tied[0][1],
                    'ranking': [m[0] for m in table]
                }
            else:
                winners[domain_lang] = {
                    'is_tie': True,
                    'winners': [{'model': m[0], 'codebleu': m[1]} for m in tied],
                    'top_codebleu': top_score,
                    'ranking': [m[0] for m in table]
                }
        return winners

    def create_overall_states(self, master_df: pd.DataFrame) -> List['AnalysisState']:
        overall_states: List[AnalysisState] = []
        base_columns = [col for col in COLUMN_NAMES.values() if col in master_df.columns]
        for language, lang_df in master_df.groupby(COLUMN_NAMES['language']):
            if pd.isna(language):
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
                # Use full language slice across all domains for this dimension
                filtered_df = lang_df[required_columns].copy()
                if filtered_df.empty:
                    continue
                context = {'results': {}, 'winners': {}}
                # Aggregate winners across domains for this language-dimension (optional, keep empty for simplicity)
                overall_states.append(AnalysisState(
                    language=str(language),
                    domain="ALL",
                    dimension=dimension,
                    data_slice=filtered_df,
                    quantitative_context=context,
                    is_overall=True
                ))
        return overall_states

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
        # Append overall language-level states
        states.extend(self.create_overall_states(master_df))
        return states

    # Build SystemAnalysisState list per (language, domain)
    def create_system_states(self, master_df: pd.DataFrame) -> List['SystemAnalysisState']:
        self.compute_system_evaluation(master_df)
        sys_states: List[SystemAnalysisState] = []
        base_columns = [col for col in COLUMN_NAMES.values() if col in master_df.columns]
        for (language, domain), group_df in master_df.groupby([COLUMN_NAMES['language'], COLUMN_NAMES['domain']]):
            if pd.isna(language) or pd.isna(domain):
                continue
            required_columns = base_columns.copy()
            # include system metric columns actually present
            present_cols: List[str] = []
            for model_name in MODEL_MAPPING.keys():
                for metric in SYSTEM_METRICS:
                    col = f"{model_name} {metric}"
                    if col in master_df.columns:
                        present_cols.append(col)
            required_columns.extend(present_cols)
            filtered_df = group_df[required_columns].copy()
            domain_key = f"{domain} ({language})"
            context = {'results': {}, 'winners': {}}
            if domain_key in self.system_results:
                context['results'] = {domain_key: self.system_results[domain_key]}
            if domain_key in self.system_winners:
                context['winners'] = {domain_key: self.system_winners[domain_key]}
            sys_states.append(SystemAnalysisState(
                language=str(language),
                domain=str(domain),
                data_slice=filtered_df,
                system_context=context
            ))
        return sys_states


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


# System EDA validator for system metrics
class SystemEDAValidator:
    def __init__(self):
        self.validation_rules = {
            'min_data_points': 1
        }

    def validate_data_slice(self, state: SystemAnalysisState) -> SystemAnalysisState:
        warnings: List[str] = []
        if state.data_slice.empty:
            state.is_data_valid = False
            state.eda_report = "❌ CRITICAL: No data found for this combination."
            state.error_message = "Empty data slice"
            return state
        if len(state.data_slice) < self.validation_rules['min_data_points']:
            warnings.append(f"Low data points: {len(state.data_slice)} rows")
        # Basic range checks for [0,1] metrics where applicable
        range_issues: List[str] = []
        for model_name in MODEL_NAMES:
            for metric in SYSTEM_METRICS:
                col = f"{model_name} {metric}"
                if col in state.data_slice.columns:
                    vals = pd.to_numeric(state.data_slice[col], errors='coerce').dropna()
                    invalid = vals[(vals < 0) | (vals > 1)]
                    if len(invalid) > 0:
                        range_issues.append(f"{model_name} {metric}: {len(invalid)} values outside [0,1]")
        warnings.extend(range_issues)
        # EDA text
        lines: List[str] = []
        lines.append(f"System-Eval Data Overview for {state.domain} ({state.language})")
        lines.append(f"   - Total data points: {len(state.data_slice)}")
        if COLUMN_NAMES['prompt_id'] in state.data_slice.columns:
            lines.append(f"   - Unique prompts: {state.data_slice[COLUMN_NAMES['prompt_id']].nunique()}")
        lines.append("\nMetric Means (by model):")
        # compute simple means per model for display
        for model_name in MODEL_NAMES:
            stats_parts: List[str] = []
            for metric in SYSTEM_METRICS:
                col = f"{model_name} {metric}"
                if col in state.data_slice.columns:
                    vals = pd.to_numeric(state.data_slice[col], errors='coerce').dropna()
                    if len(vals) > 0:
                        stats_parts.append(f"{metric}={vals.mean():.3f}")
            if stats_parts:
                lines.append(f"   - {MODEL_DISPLAY_NAMES[model_name]}: " + ", ".join(stats_parts))
        if warnings:
            lines.append("\nValidation Warnings:")
            for w in warnings:
                lines.append(f"   - {w}")
        state.is_data_valid = True
        state.validation_warnings = warnings
        state.eda_report = '\n'.join(lines)
        return state


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
            if state.is_overall:
                prompt = self._create_overall_prompt(state, csv_string)
            else:
                prompt = self._create_enhanced_prompt(state, csv_string)
            response = self.model.generate_content(prompt)
            full_text = (response.text or "").strip()
            state.analysis_report = full_text
            if state.is_overall:
                # Extract compact winner sentence and Gemini-only overall paragraph
                winner_line, gemini_overall = self._extract_overall_sections(full_text)
                state.winner_text = winner_line
                state.client_performance_text = gemini_overall
                return state
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

    def _create_overall_prompt(self, state: AnalysisState, csv_data: str) -> str:
        # Overall prompt distilled from the provided example and requirements
        return f"""
You are assisting in a human evaluation project comparing OpenAI o4-mini-high, Gemini 2.5 pro, and Claude Opus 4.
We have collected ratings for the '{state.dimension}' dimension from multiple evaluators across multiple topics.
Analyze the CSV for ALL industries within Language='{state.language}'. Do not reference Prompt IDs.

Tasks (do not repeat or restate these instructions in your output):
1) For each Industry present in the CSV (within Language='{state.language}'), identify the TOP performer (most number of 5s for '{state.dimension}'). For each industry, write ONE very short sentence: "[Industry]: The top performer is [MODEL_DISPLAY_NAME]." Optionally add one short justification phrase.
2) Then write EXACTLY ONE short paragraph (3-4 lines) describing the overall performance of Gemini 2.5 pro across all industries for Language='{state.language}' and '{state.dimension}'. No bullets. Mention one topic Gemini is good at and one topic it is bad at. Do not reference prompt IDs.
3) Keep everything concise. No extra sections. Begin directly with the industry sentences, followed by the single Gemini paragraph.

CSV:
```csv
{csv_data}
```
"""

    def _extract_overall_sections(self, full_text: str) -> tuple[str, str]:
        # Winner line: first non-empty line(s) until a blank, then Gemini paragraph
        lines = [l.strip() for l in full_text.splitlines()]
        lines = [l for l in lines if l]
        winner_lines: List[str] = []
        gemini_paragraph = ""
        # Collect industry lines until we hit a line starting with 'Overall' or 'Gemini'
        i = 0
        while i < len(lines) and not lines[i].lower().startswith("overall") and not lines[i].lower().startswith("gemini"):
            winner_lines.append(lines[i])
            i += 1
        # Remaining lines form the Gemini paragraph; join and then filter to only the Gemini overall part if prefixed by Overall,/Gemini
        remaining = " ".join(lines[i:]).strip()
        # Keep only the paragraph about Gemini (heuristic: start at first 'Overall, Gemini' or 'Gemini 2.5 pro')
        lowered = remaining.lower()
        start_idx = max(lowered.find("overall, gemini"), lowered.find("overall gemini"), lowered.find("gemini 2.5 pro"))
        if start_idx != -1:
            gemini_paragraph = remaining[start_idx:].strip()
        else:
            gemini_paragraph = remaining
        # Winner line collapsed to single line
        winner_line = " ".join(winner_lines).strip()
        return winner_line, gemini_paragraph


# System analysis agent for CodeBLEU-style narrative
class SystemAnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def generate_analysis(self, state: SystemAnalysisState) -> SystemAnalysisState:
        try:
            with io.StringIO() as buffer:
                state.data_slice.to_csv(buffer, index=False)
                csv_string = buffer.getvalue()
            prompt = self._create_system_prompt(state, csv_string)
            response = self.model.generate_content(prompt)
            full_text = (response.text or "").strip()
            state.analysis_report = full_text
            # Parse into winner vs Gemini overview if marker exists
            client_header = f"Overall assessment of {CLIENT_MODEL_NAME}"
            idx = full_text.lower().find(client_header.lower())
            if idx != -1:
                state.winner_text = full_text[:idx].strip()
                state.client_performance_text = full_text[idx:].strip()
            else:
                state.winner_text = full_text
                state.client_performance_text = ""
            return state
        except Exception as e:
            state.error_message = f"Error generating system analysis: {e}"
            state.analysis_report = state.error_message
            return state

    def _create_system_prompt(self, state: SystemAnalysisState, csv_data: str) -> str:
        domain_key = f"{state.domain} ({state.language})"
        # Build metric summary from context
        summary_lines: List[str] = []
        winners_line = "No winner information available."
        res = state.system_context.get('results', {}).get(domain_key, {}).get('metrics', {})
        win = state.system_context.get('winners', {}).get(domain_key, {})
        for model_name in MODEL_NAMES:
            if model_name in res:
                parts: List[str] = []
                for metric in SYSTEM_METRICS:
                    if metric in res[model_name]:
                        parts.append(f"{metric}={res[model_name][metric]['mean']:.3f}")
                if parts:
                    summary_lines.append(f"- {MODEL_DISPLAY_NAMES[model_name]}: " + ", ".join(parts))
        if win:
            if win.get('is_tie'):
                winners_line = "Winners (tie): " + ", ".join([MODEL_DISPLAY_NAMES[w['model']] for w in win.get('winners', [])]) + f" (CodeBLEU={win.get('top_codebleu'):.3f})"
            else:
                winners_line = f"Winner: {MODEL_DISPLAY_NAMES[win['winner']]} (CodeBLEU={win.get('winner_codebleu'):.3f})"

        metric_summary = "\n".join(summary_lines) if summary_lines else "No metric summary available."

        return f"""
You are evaluating system-level code quality metrics (CodeBLEU and related sub-scores) for the '{state.domain}' industry in '{state.language}'.

Context (aggregated means by model):
{metric_summary}

{winners_line}

CSV (slice for this industry/language):
```csv
{csv_data}
```

Your Task:
1) Write 2-3 concise sentences declaring which model leads on CodeBLEU in this industry/language and why, citing specific metric means (e.g., N-gram, Dataflow) from the context above.
2) Then write ONE short paragraph starting with exactly: "Overall assessment of {CLIENT_MODEL_NAME}:" that describes {CLIENT_MODEL_NAME}'s system performance in this industry, highlighting one strength and one weakness based on the metrics.
3) Keep it focused, professional, and evidence-based. No bullet points.
"""


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


# System analysis workflow (validate -> analysis -> end)

def create_system_analysis_workflow():
    eda = SystemEDAValidator()
    analyst = SystemAnalysisAgent()

    def eda_node(s: SystemAnalysisState) -> SystemAnalysisState:
        return eda.validate_data_slice(s)

    def analysis_node(s: SystemAnalysisState) -> SystemAnalysisState:
        return analyst.generate_analysis(s)

    workflow = StateGraph(SystemAnalysisState)
    workflow.add_node("eda", eda_node)
    workflow.add_node("analysis", analysis_node)
    workflow.set_entry_point("eda")
    workflow.add_edge("eda", "analysis")
    workflow.add_edge("analysis", END)
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
        self.overall_entries: List[Dict[str, Any]] = []
        # System entries
        self.system_quant_entries: List[Dict[str, Any]] = []
        self.system_qual_entries: List[Dict[str, Any]] = []

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
                # Separate topics if present
                topics_payload = {}
                if 'topics' in models:
                    topics_payload = models['topics']
                    models = {k: v for k, v in models.items() if k != 'topics'}
                entries.append({
                    'analysis_id': analysis_id,
                    'language': language,
                    'domain': domain,
                    'dimension': dimension,
                    'models': models,
                    'topics': topics_payload,
                    'winner': winner_info
                })
        self.quantitative_entries = entries

    def collect_qualitative(self, completed_states: List[AnalysisState]):
        q_entries: List[Dict[str, Any]] = []
        overall_entries: List[Dict[str, Any]] = []
        for s in completed_states:
            analysis_id = make_analysis_id(s.language, s.domain, s.dimension)
            entry = {
                'analysis_id': analysis_id,
                'language': s.language,
                'domain': s.domain,
                'dimension': s.dimension,
                'winner_text': s.winner_text,
                'client_performance_text': s.client_performance_text,
                'is_error': bool(s.error_message)
            }
            if getattr(s, 'is_overall', False):
                overall_entries.append(entry)
            else:
                q_entries.append(entry)
        self.qualitative_entries = q_entries
        self.overall_entries = overall_entries

    # System aggregates collection
    def collect_system_quantitative(self):
        entries: List[Dict[str, Any]] = []
        for domain_lang, payload in self.slicer.system_results.items():
            try:
                domain, lang_part = domain_lang.split(' (')
                language = lang_part[:-1]
            except Exception:
                parts = domain_lang.split('|')
                domain = parts[0] if parts else domain_lang
                language = parts[1] if len(parts) > 1 else "Unknown"
            analysis_id = make_analysis_id(language, domain, 'SYSTEM')
            winner_info = self.slicer.system_winners.get(domain_lang, {})
            entries.append({
                'analysis_id': analysis_id,
                'language': language,
                'domain': domain,
                'metrics': payload.get('metrics', {}),
                'winner': winner_info
            })
        self.system_quant_entries = entries

    def collect_system_qualitative(self, completed_states: List[SystemAnalysisState]):
        entries: List[Dict[str, Any]] = []
        for s in completed_states:
            analysis_id = make_analysis_id(s.language, s.domain, 'SYSTEM')
            entries.append({
                'analysis_id': analysis_id,
                'language': s.language,
                'domain': s.domain,
                'winner_text': s.winner_text,
                'client_performance_text': s.client_performance_text,
                'is_error': bool(s.error_message)
            })
        self.system_qual_entries = entries

    def save_json_contracts(self):
        ensure_dirs()
        with open(OUTPUT_QUANTITATIVE_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.quantitative_entries, f, indent=2, ensure_ascii=False)
        with open(OUTPUT_QUALITATIVE_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.qualitative_entries, f, indent=2, ensure_ascii=False)
        with open(OUTPUT_OVERALL_QUAL_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.overall_entries, f, indent=2, ensure_ascii=False)
        # System outputs
        with open(OUTPUT_SYSTEM_QUANT_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.system_quant_entries, f, indent=2, ensure_ascii=False)
        with open(OUTPUT_SYSTEM_QUAL_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.system_qual_entries, f, indent=2, ensure_ascii=False)
        print(f"Wrote quantitative -> {OUTPUT_QUANTITATIVE_JSON}")
        print(f"Wrote qualitative -> {OUTPUT_QUALITATIVE_JSON}")
        print(f"Wrote overall qualitative -> {OUTPUT_OVERALL_QUAL_JSON}")
        print(f"Wrote system quantitative -> {OUTPUT_SYSTEM_QUANT_JSON}")
        print(f"Wrote system qualitative -> {OUTPUT_SYSTEM_QUAL_JSON}")

    # Step 9.1 (Why): Emit CSVs for system evals to simplify verification/graphing
    # Step 9.1 (Done): Added CSV writers for system quantitative and qualitative data
    def save_system_csvs(self) -> None:
        ensure_dirs()
        # Build quantitative long-format CSV rows
        quant_rows: List[Dict[str, Any]] = []
        for domain_lang, payload in self.slicer.system_results.items():
            try:
                domain, lang_part = domain_lang.split(' (')
                language = lang_part[:-1]
            except Exception:
                parts = domain_lang.split('|')
                domain = parts[0] if parts else domain_lang
                language = parts[1] if len(parts) > 1 else "Unknown"
            winner_info = self.slicer.system_winners.get(domain_lang, {})
            is_tie = bool(winner_info.get('is_tie', False)) if winner_info else False
            winner_models: List[str] = []
            winner_codebleu: float | None = None
            if winner_info:
                if is_tie:
                    winner_models = [w.get('model') for w in winner_info.get('winners', [])]
                    winner_codebleu = winner_info.get('top_codebleu')
                else:
                    winner_models = [winner_info.get('winner')]
                    winner_codebleu = winner_info.get('winner_codebleu')
            metrics_block = payload.get('metrics', {})
            for model_key, metrics in metrics_block.items():
                row: Dict[str, Any] = {
                    'Code Language': language,
                    'Industry': domain,
                    'Model': MODEL_DISPLAY_NAMES.get(model_key, model_key),
                    'Model Key': model_key,
                    'Is Winner': model_key in set(winner_models),
                    'Is Tie': is_tie,
                    'Winner CodeBLEU': winner_codebleu if winner_codebleu is not None else ''
                }
                # Flatten metric stats
                for metric_name in SYSTEM_METRICS:
                    stat = metrics.get(metric_name)
                    if stat:
                        row[f"{metric_name} Mean"] = stat.get('mean')
                        row[f"{metric_name} Std"] = stat.get('std')
                        row[f"{metric_name} Count"] = stat.get('count')
                    else:
                        row[f"{metric_name} Mean"] = ''
                        row[f"{metric_name} Std"] = ''
                        row[f"{metric_name} Count"] = ''
                quant_rows.append(row)
        if quant_rows:
            pd.DataFrame(quant_rows).to_csv(OUTPUT_SYSTEM_QUANT_CSV, index=False)
            print(f"Wrote system quantitative CSV -> {OUTPUT_SYSTEM_QUANT_CSV}")
        else:
            print("⚠️ No system quantitative data to write CSV.")

        # Qualitative CSV rows from collected entries
        if self.system_qual_entries:
            qual_rows = []
            for e in self.system_qual_entries:
                qual_rows.append({
                    'Code Language': e.get('language'),
                    'Industry': e.get('domain'),
                    'Winner Text': e.get('winner_text', ''),
                    'Client Performance Text': e.get('client_performance_text', ''),
                    'Is Error': e.get('is_error', False)
                })
            pd.DataFrame(qual_rows).to_csv(OUTPUT_SYSTEM_QUAL_CSV, index=False)
            print(f"Wrote system qualitative CSV -> {OUTPUT_SYSTEM_QUAL_CSV}")
        else:
            print("⚠️ No system qualitative entries to write CSV.")

    # Step 9.2 (Why): Emit an RLHF-style per-prompt system eval CSV matching the provided sample structure
    # Columns: Code Language, Prompt ID, Domain, Subtopic, <Display> Codebleu Score (per model)
    def write_rlhf_system_evals_csv(self, master_df: pd.DataFrame) -> None:
        ensure_dirs()
        lang_col = COLUMN_NAMES['language']
        pid_col = COLUMN_NAMES['prompt_id']
        ind_col = COLUMN_NAMES['domain']
        topic_col = COLUMN_NAMES['topic']
        # Required structure must exist; otherwise skip gracefully
        missing = [c for c in [lang_col, pid_col, ind_col, topic_col] if c not in master_df.columns]
        if missing:
            print(f"⚠️ Skipping RLHF system eval CSV; missing base columns: {missing}")
            return
        # Build rows
        rows: List[Dict[str, Any]] = []
        for _, row in master_df.iterrows():
            out_row: Dict[str, Any] = {
                'Code Language': row.get(lang_col),
                'Prompt ID': row.get(pid_col),
                'Domain': row.get(ind_col),
                'Subtopic': row.get(topic_col)
            }
            # Per-model Codebleu Score columns in master: e.g., 'Chatgpt Codebleu Score'
            for model_key in MODEL_NAMES:
                source_col = f"{model_key} Codebleu Score"
                display_name = SYSTEM_CSV_DISPLAY_NAMES.get(model_key, model_key)
                dest_col = f"{display_name} Codebleu Score"
                out_row[dest_col] = row.get(source_col) if source_col in master_df.columns else ''
            rows.append(out_row)
        pd.DataFrame(rows).to_csv(OUTPUT_RLHF_SYSTEM_EVALS_CSV, index=False)
        print(f"Wrote RLHF system evals CSV -> {OUTPUT_RLHF_SYSTEM_EVALS_CSV}")

    def write_topic_csvs(self, master_df: pd.DataFrame) -> None:
        """Generate a helper CSV grouped by Topic similar to RLHF_data_analyzed_by_topic.csv (wide format only)."""
        ensure_dirs()
        # Determine models/dimensions present
        models_present: List[str] = []
        for m in MODEL_NAMES:
            for d in DIMENSIONS_TO_ANALYZE:
                if f"{m} Human {d}" in master_df.columns:
                    models_present.append(m)
                    break
        models_present = list(dict.fromkeys(models_present))  # dedupe preserve order

        # Column names
        lang_col = COLUMN_NAMES['language']
        ind_col = COLUMN_NAMES['domain']
        topic_col = COLUMN_NAMES['topic']
        pid_col = COLUMN_NAMES['prompt_id']
        required_cols = [c for c in [lang_col, ind_col, topic_col, pid_col] if c in master_df.columns]
        if len(required_cols) < 4:
            # Missing structural columns; skip
            return

        rows_wide: List[Dict[str, Any]] = []
        for (language, industry, topic, prompt_id), gdf in master_df.groupby([lang_col, ind_col, topic_col, pid_col]):
            row = {
                'Code Language': language,
                'Industry': industry,
                'Topic': topic,
                'Prompt ID': prompt_id,
            }
            for m in models_present:
                for d in DIMENSIONS_TO_ANALYZE:
                    col = f"{m} Human {d}"
                    header = f"{m} - {d} (count of 5s)"
                    count5 = int((gdf[col] == 5).sum()) if col in gdf.columns else 0
                    row[header] = count5
            rows_wide.append(row)

        # Write wide file only
        wide_path = os.path.join(DATA_DIR, 'RLHF_data_analyzed_by_topic.csv')
        pd.DataFrame(rows_wide).to_csv(wide_path, index=False)
        print(f"Wrote helper CSV -> {wide_path}")


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

    # Create states (human eval)
    slicer = DataSlicerTool()
    states = slicer.create_analysis_states(master_df)
    if not states:
        print("❌ No analysis states created (human eval). Exiting.")
        return

    # Compile workflows
    workflow = create_analysis_workflow()
    system_workflow = create_system_analysis_workflow()

    # Execute human analyses in parallel
    completed: List[AnalysisState] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.optimal_workers) as executor:
        futures = [executor.submit(lambda st: workflow.invoke(st), s) for s in states]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if isinstance(res, dict):
                s = AnalysisState(
                    language=res.get('language', 'Unknown'),
                    domain=res.get('domain', 'Unknown'),
                    dimension=res.get('dimension', 'Unknown'),
                    data_slice=pd.DataFrame(),
                )
                s.is_overall = bool(res.get('is_overall', False))
                s.eda_report = res.get('eda_report', '')
                s.analysis_report = res.get('analysis_report', '')
                s.error_message = res.get('error_message', '')
                s.winner_text = res.get('winner_text', '')
                s.client_performance_text = res.get('client_performance_text', '')
                completed.append(s)
            else:
                completed.append(res)

    # Prepare and execute system analyses (per language/domain)
    system_states = slicer.create_system_states(master_df)
    completed_system: List[SystemAnalysisState] = []
    if system_states:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.optimal_workers) as executor:
            futures = [executor.submit(lambda st: system_workflow.invoke(st), s) for s in system_states]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if isinstance(res, dict):
                    s = SystemAnalysisState(
                        language=res.get('language', 'Unknown'),
                        domain=res.get('domain', 'Unknown'),
                        data_slice=pd.DataFrame(),
                    )
                    s.eda_report = res.get('eda_report', '')
                    s.analysis_report = res.get('analysis_report', '')
                    s.error_message = res.get('error_message', '')
                    s.winner_text = res.get('winner_text', '')
                    s.client_performance_text = res.get('client_performance_text', '')
                    completed_system.append(s)
                else:
                    completed_system.append(res)
    else:
        print("⚠️ No system evaluation states created (missing CodeBLEU columns?). Skipping system analysis.")

    # Aggregate and save contracts
    agg = ReportAggregator(slicer)
    agg.collect_quantitative()
    agg.collect_qualitative(completed)
    # System aggregates
    agg.collect_system_quantitative()
    agg.collect_system_qualitative(completed_system)
    agg.save_json_contracts()
    # Also emit CSV files for system evals
    try:
        agg.save_system_csvs()
    except Exception as e:
        print(f"⚠️ Failed writing system CSVs: {e}")
    # Write RLHF-style system eval CSV (per-prompt)
    try:
        agg.write_rlhf_system_evals_csv(master_df)
    except Exception as e:
        print(f"⚠️ Skipped writing RLHF system eval CSV due to error: {e}")
    # Write helper CSVs for graph building
    try:
        agg.write_topic_csvs(master_df)
    except Exception as e:
        print(f"⚠️ Skipped writing helper topic CSVs due to error: {e}")
    print("🎉 analysis_generator complete")


if __name__ == "__main__":
    main()      