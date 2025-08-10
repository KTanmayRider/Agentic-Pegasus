import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
import google.generativeai as genai
import re
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

plt.style.use('default')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyD142bMLF0FPWStCKpovbUYTC10vIhO1q8"
genai.configure(api_key=GEMINI_API_KEY)

# Model Display Name Mapping
MODEL_DISPLAY_NAMES = {
    'Chatgpt': 'OpenAI o4-mini-high',
    'Gemini': 'Gemini 2.5 pro',
    'Claude': 'Claude Opus 4'
}


def detect_models_with_gemini(df_columns: List[str]) -> Dict[str, str]:
    """Use Gemini API to detect model names from CSV columns"""
    
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analyze these CSV column names and identify the AI model names mentioned:
        {df_columns}
        
        Return a JSON object mapping each detected model to these specific display names:
        - "OpenAI o4-mini-high" for any OpenAI/GPT/ChatGPT models
        - "Gemini 2.5 pro" for any Google/Gemini models  
        - "Claude Opus 4" for any Anthropic/Claude models
        
        Example output format:
        {{
            "Chatgpt": "OpenAI o4-mini-high",
            "Gemini": "Gemini 2.5 pro", 
            "Claude": "Claude Opus 4"
        }}
        
        Only return the JSON object, no other text.
        """
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        import json
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
            
        model_mapping = json.loads(response_text)
        print(f"✅ Gemini API detected models:")
        for original, standard in model_mapping.items():
            print(f"   • {original} → {standard}")
        return model_mapping
        
    except Exception as e:
        print(f"⚠️ Gemini API failed: {e}. Using fallback detection.")
        return detect_models_fallback(df_columns)

def detect_models_fallback(df_columns: List[str]) -> Dict[str, str]:
    """Fallback model detection using pattern matching"""
    model_mapping = {}
    
    for col in df_columns:
        col_lower = col.lower()
        if 'chatgpt' in col_lower or 'gpt' in col_lower or 'openai' in col_lower:
            # Extract the actual model name from column
            if 'chatgpt' in col_lower:
                model_mapping['Chatgpt'] = MODEL_DISPLAY_NAMES['Chatgpt']
        elif 'gemini' in col_lower or 'google' in col_lower:
            model_mapping['Gemini'] = MODEL_DISPLAY_NAMES['Gemini']
        elif 'claude' in col_lower or 'anthropic' in col_lower:
            model_mapping['Claude'] = MODEL_DISPLAY_NAMES['Claude']
    
    print(f"✅ Fallback pattern matching detected models:")
    for original, display_name in model_mapping.items():
        print(f"   • {original} → {display_name}")
    return model_mapping

def load_clean_and_sort_data(filepath):
    """Load, clean and sort human evaluation data"""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{filepath}' was not found.")
        return None, {}

    # Detect models dynamically
    count_columns = [col for col in df.columns if '(count of 5s)' in col]
    model_mapping = detect_models_with_gemini(count_columns)
    
    df['Code Language'] = df['Code Language'].str.lower()
    df['Prompt ID'] = pd.to_numeric(df['Prompt ID'], errors='coerce')
    df.dropna(subset=['Prompt ID'], inplace=True)
    df['Prompt ID'] = df['Prompt ID'].astype(int)

    for col in count_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['Topic'] = df['Topic'].str.replace('‑', ' ', regex=False)
    df.sort_values(by='Prompt ID', inplace=True)
    return df, model_mapping

def create_and_save_canvas_plot(detailed_df, overall_df, metric_name, filename, model_mapping):
    """Generate 4x4 canvas plot with dynamic model detection"""
    fig, axs = plt.subplots(4, 4, figsize=(24, 22))
    fig.suptitle(f'Human Evaluations - Metric: {metric_name}', fontsize=28, y=0.98)

    # Dynamic color mapping based on detected models
    default_colors = ['#FF8080', '#80FF80', '#8080FF', '#FFB366', '#FF66B3']
    model_names = list(model_mapping.values())
    colors = {model_names[i]: default_colors[i % len(default_colors)] for i in range(len(model_names))}
    
    # Get available languages from data
    available_languages = detailed_df['Code Language'].unique()
    language_order = []
    lang_code_map = {}
    
    # Map common language codes
    lang_mappings = {
        'python': ('Python', 'python'),
        'cpp': ('C++', 'cpp'), 
        'c++': ('C++', 'cpp'),
        'java': ('Java', 'java'),
        'javascript': ('JavaScript', 'javascript'),
        'js': ('JavaScript', 'javascript')
    }
    
    for lang_code in available_languages:
        if lang_code.lower() in lang_mappings:
            display_name, code = lang_mappings[lang_code.lower()]
            language_order.append(display_name)
            lang_code_map[display_name] = code
        else:
            # Fallback for unknown languages
            display_name = lang_code.capitalize()
            language_order.append(display_name)
            lang_code_map[display_name] = lang_code.lower()

    # Dynamic language colors
    language_colors = ['#00FFFF', '#3776AB', '#ed8b00', '#f7df1e', '#ff6b6b', '#4ecdc4']
    language_label_colors = {lang: language_colors[i % len(language_colors)] for i, lang in enumerate(language_order)}

    # Dynamic column mapping based on detected models
    model_columns = {}
    for original_name, standard_name in model_mapping.items():
        col_pattern = f'{original_name} - {metric_name} (count of 5s)'
        if col_pattern in detailed_df.columns:
            model_columns[standard_name] = col_pattern

    for row_idx, lang_name in enumerate(language_order[:4]):  # Limit to 4 rows
        lang_code = lang_code_map[lang_name]
        lang_overall_data = overall_df[overall_df['Code Language'] == lang_code]
        lang_detailed_data = detailed_df[detailed_df['Code Language'] == lang_code]

        # Calculate common y limit dynamically
        row_max_y = 0
        if not lang_overall_data.empty:
            available_cols = [col for col in model_columns.values() if col in lang_overall_data.columns]
            if available_cols:
                row_max_y = max(row_max_y, lang_overall_data[available_cols].max().max())
        if not lang_detailed_data.empty:
            available_cols = [col for col in model_columns.values() if col in lang_detailed_data.columns]
            if available_cols:
                row_max_y = max(row_max_y, lang_detailed_data[available_cols].max().max())
        common_y_limit = max(row_max_y + 2, 5)

        # Overall Chart (Column 0)
        ax_overall = axs[row_idx, 0]
        if not lang_overall_data.empty:
            industries = lang_overall_data['Industry'].tolist()
            x = np.arange(len(industries))
            width = 0.8 / len(model_columns)  # Dynamic width based on number of models

            # Plot bars for each model dynamically
            for i, (model_name, col_name) in enumerate(model_columns.items()):
                if col_name in lang_overall_data.columns:
                    offset = (i - len(model_columns)/2 + 0.5) * width
                    ax_overall.bar(x + offset, lang_overall_data[col_name], width, 
                                 color=colors[model_name], label=model_name)

            lang_color = language_label_colors[lang_name]
            ax_overall.set_title('Overall', y=1.05, fontweight='bold', color='black', fontsize=16,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=lang_color, alpha=0.7))

            wrapped_labels = [textwrap.fill(l, 12) for l in industries]
            ax_overall.set_xticks(x)
            ax_overall.set_xticklabels(wrapped_labels, fontsize=10, ha='center')
            ax_overall.set_ylim(0, common_y_limit)
            ax_overall.set_yticks(np.arange(0, common_y_limit, step=max(1, int(common_y_limit/5))))
            ax_overall.grid(axis='y', linestyle='--', alpha=0.7)

        # Detailed Industry Charts (Columns 1, 2, 3)
        industries_in_lang = lang_overall_data['Industry'].tolist()

        for col_offset, industry_name in enumerate(industries_in_lang[:3]):
            ax = axs[row_idx, col_offset + 1]
            industry_slice = lang_detailed_data[lang_detailed_data['Industry'] == industry_name]

            if not industry_slice.empty:
                subtopics = industry_slice['Topic'].tolist()
                x = np.arange(len(subtopics))
                width = 0.8 / len(model_columns)  # Dynamic width

                # Plot bars for each model dynamically
                for i, (model_name, col_name) in enumerate(model_columns.items()):
                    if col_name in industry_slice.columns:
                        offset = (i - len(model_columns)/2 + 0.5) * width
                        ax.bar(x + offset, industry_slice[col_name], width, 
                             color=colors[model_name])

            lang_color = language_label_colors[lang_name]
            ax.set_title(industry_name, y=1.05, fontweight='bold', color='black', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=lang_color, alpha=0.7))

            wrapped_subtopics = [textwrap.fill(s, 15) for s in subtopics]
            ax.set_xticks(x)
            ax.set_xticklabels(wrapped_subtopics, rotation=0, ha='center', fontsize=8)
            ax.set_ylim(0, common_y_limit)
            ax.set_yticks(np.arange(0, common_y_limit, step=max(1, int(common_y_limit/5))))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add axis labels
    for row_idx, lang_name in enumerate(language_order):
        axs[row_idx, 0].set_xlabel('Industry', fontsize=12, fontweight='bold')
        for col_idx in range(1, 4):
            if axs[row_idx, col_idx].has_data():
                axs[row_idx, col_idx].set_xlabel('Topic', fontsize=10, fontweight='bold')

    # Hide empty subplots
    for ax in axs.flat:
        if not ax.has_data():
            ax.set_visible(False)

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.96], h_pad=3.0)

    # Language labels
    for row_idx, lang_name in enumerate(language_order):
        pos = axs[row_idx, 0].get_position()
        y_fig_coord = pos.y0 + pos.height / 2
        x_fig_coord = 0.04
        lang_box_color = language_label_colors[lang_name]
        fig.text(x_fig_coord, y_fig_coord, lang_name, ha='center', va='center', rotation=90,
                fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc=lang_box_color, ec="black", lw=1, alpha=0.5))

    # Legend and chart guide
    handles = [plt.Rectangle((0,0),1,1, color=colors[model_name]) for model_name in model_names]
    shared_y_coord = 0.04

    legend = fig.legend(handles, model_names,
                    loc='center', bbox_to_anchor=(0.35, shared_y_coord),
                    ncol=len(model_names), title='Models', title_fontsize=14, fontsize=12,
                    frameon=True, fancybox=True, shadow=True,
                    edgecolor='gray', framealpha=1, facecolor='white')
    legend.get_frame().set_linewidth(1.5)

    guide_text_raw = "Chart Guide\nX-Axis: Count of '5' Ratings"
    fig.text(0.65, shared_y_coord, guide_text_raw, ha='center', va='center',
            fontsize=12, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', lw=1.5))

    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"✅ Canvas saved: {filename}")

# EXECUTE: Generate 4x4 Human Evaluation Canvas Plots
print("🎨 Generating 4x4 Human Evaluation Canvas Plots...")

csv_filepath = 'output/data/RLHF_data_analyzed_by_topic.csv'
df, model_mapping = load_clean_and_sort_data(csv_filepath)

if df is not None and model_mapping:
    output_dir = "output/graphs/dynamic_human_evals"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All canvas graphs will be saved in: '{output_dir}'")

    count_columns = [col for col in df.columns if '(count of 5s)' in col]
    agg_operations = {col: 'sum' for col in count_columns}
    agg_operations['Prompt ID'] = 'min'

    overall_agg_df = df.groupby(['Code Language', 'Industry']).agg(agg_operations).reset_index()
    overall_agg_df.sort_values(by='Prompt ID', inplace=True)

    metrics_to_plot = ['Completeness', 'Correctness', 'Relevance']

    for metric in metrics_to_plot:
        print(f"\n--- Generating canvas for Metric: '{metric}' ---")
        filename = f"Dynamic_Canvas_{metric}.png"
        full_path = os.path.join(output_dir, filename)
        create_and_save_canvas_plot(df, overall_agg_df, metric, full_path, model_mapping)
else:
    print("❌ Could not load data. Check file path.")
def create_industry_plot(language, industry, data_slice, metric_name, filename, model_mapping):
    """Create individual industry plot with dynamic model detection"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Dynamic color mapping
    default_colors = ['#FF8080', '#80FF80', '#8080FF', '#FFB366', '#FF66B3']
    model_names = list(model_mapping.values())
    colors = {model_names[i]: default_colors[i % len(default_colors)] for i in range(len(model_names))}

    subtopics = data_slice['Topic'].tolist()
    x = np.arange(len(subtopics))
    width = 0.8 / len(model_mapping)  # Dynamic width

    # Dynamic column mapping
    model_columns = {}
    for original_name, standard_name in model_mapping.items():
        col_pattern = f'{original_name} - {metric_name} (count of 5s)'
        if col_pattern in data_slice.columns:
            model_columns[standard_name] = col_pattern

    # Plot bars for each model dynamically
    for i, (model_name, col_name) in enumerate(model_columns.items()):
        if col_name in data_slice.columns:
            offset = (i - len(model_columns)/2 + 0.5) * width
            counts = data_slice[col_name].tolist()
            ax.bar(x + offset, counts, width, label=model_name, color=colors[model_name])

    ax.set_ylabel("Count of '5' Ratings", fontsize=12, fontweight='bold')
    ax.set_xlabel("Topics", fontsize=12, fontweight='bold')
    ax.set_title(f'{language} - {industry}', fontsize=16, fontweight='bold')

    # Calculate max count dynamically from all model data
    max_count = 0
    for col_name in model_columns.values():
        if col_name in data_slice.columns:
            max_count = max(max_count, data_slice[col_name].max())
    
    y_limit = max(max_count + 1, 3)
    ax.set_ylim(0, y_limit)
    ax.set_yticks(np.arange(0, y_limit + 1, 1))

    wrapped_subtopics = [textwrap.fill(s, 20) for s in subtopics]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_subtopics, fontsize=9)
    ax.tick_params(axis='x', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Models', fontsize=11)

    fig.suptitle(f'Evaluation Metric: {metric_name}', fontsize=14, color='black')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"✅ Saved: {filename}")

def create_overall_plot(language, overall_data_slice, metric_name, filename, model_mapping):
    """Create overall language plot with dynamic model detection"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Dynamic color mapping
    default_colors = ['#FF8080', '#80FF80', '#8080FF', '#FFB366', '#FF66B3']
    model_names = list(model_mapping.values())
    colors = {model_names[i]: default_colors[i % len(default_colors)] for i in range(len(model_names))}

    industries = overall_data_slice['Industry'].tolist()
    x = np.arange(len(industries))
    width = 0.8 / len(model_mapping)  # Dynamic width

    # Dynamic column mapping
    model_columns = {}
    for original_name, standard_name in model_mapping.items():
        col_pattern = f'{original_name} - {metric_name} (count of 5s)'
        if col_pattern in overall_data_slice.columns:
            model_columns[standard_name] = col_pattern

    # Plot bars for each model dynamically
    for i, (model_name, col_name) in enumerate(model_columns.items()):
        if col_name in overall_data_slice.columns:
            offset = (i - len(model_columns)/2 + 0.5) * width
            counts = overall_data_slice[col_name].tolist()
            ax.bar(x + offset, counts, width, label=model_name, color=colors[model_name])

    ax.set_ylabel("Total Count of '5' Ratings (All Topics)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Industries", fontsize=12, fontweight='bold')
    ax.set_title(f'{language} - Overall', fontsize=16, fontweight='bold')

    # Calculate max count dynamically from all model data
    max_count = 0
    for col_name in model_columns.values():
        if col_name in overall_data_slice.columns:
            max_count = max(max_count, overall_data_slice[col_name].max())
    
    y_limit = max(max_count + 1, 5)
    ax.set_ylim(0, y_limit)
    ax.set_yticks(np.arange(0, y_limit + 1, step=max(1, int(y_limit/5))))

    wrapped_industries = [textwrap.fill(s, 18) for s in industries]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_industries, fontsize=10)
    ax.tick_params(axis='x', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Models', fontsize=11)

    fig.suptitle(f'Evaluation Metric: {metric_name}', fontsize=14, color='black')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"✅ Saved: {filename}")

# EXECUTE: Generate Individual Human Evaluation Plots
print("📈 Generating Individual Human Evaluation Plots...")

csv_filepath = 'output/data/RLHF_data_analyzed_by_topic.csv'
df, model_mapping = load_clean_and_sort_data(csv_filepath)

if df is not None and model_mapping:
    output_dir_industry = "output/graphs/Individual_dynamic_human_evals"
    output_dir_overall = "output/graphs/Overall_dynamic_human_evals"
    os.makedirs(output_dir_industry, exist_ok=True)
    os.makedirs(output_dir_overall, exist_ok=True)
    print(f"Detailed graphs will be saved in: '{output_dir_industry}'")
    print(f"Overall graphs will be saved in: '{output_dir_overall}'")

    count_columns = [col for col in df.columns if '(count of 5s)' in col]
    agg_logic = {col: 'sum' for col in count_columns}
    agg_logic['Prompt ID'] = 'min'
    overall_agg_df = df.groupby(['Code Language', 'Industry']).agg(agg_logic).reset_index()
    overall_agg_df.sort_values(by='Prompt ID', inplace=True)

    metrics_to_plot = ['Completeness', 'Correctness', 'Relevance']

    for metric in metrics_to_plot:
        print(f"\n--- Generating graphs for Metric: '{metric}' ---")

        # Generate Industry-Specific Graphs
        print(" Generating detailed industry-topic graphs...")
        industry_grouped = df.groupby(['Code Language', 'Industry'], sort=False)

        for (lang, industry), group_df in industry_grouped:
            display_lang = 'C++' if lang.lower() == 'cpp' else lang.capitalize()
            filename = f"Dynamic_{metric}_{display_lang.replace('C++', 'Cpp')}-{industry.replace(' ', '_')}.png"
            full_path = os.path.join(output_dir_industry, filename)
            create_industry_plot(display_lang, industry, group_df, metric, full_path, model_mapping)

        # Generate Overall Language Graphs
        print(" Generating overall language graphs...")
        overall_grouped = overall_agg_df.groupby('Code Language', sort=False)

        for lang, lang_group_df in overall_grouped:
            display_lang = 'C++' if lang.lower() == 'cpp' else lang.capitalize()
            filename = f"Dynamic_{metric}_{display_lang.replace('C++', 'Cpp')}-Overall.png"
            full_path = os.path.join(output_dir_overall, filename)
            create_overall_plot(display_lang, lang_group_df, metric, full_path, model_mapping)
else:
    print("❌ Could not load data. Check file path.")
