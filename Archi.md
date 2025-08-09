### **Detailed Agentic Architecture for the 3-File Reporting Pipeline**

This architecture outlines three independent Python scripts. They are executed sequentially, with the outputs of one script serving as the inputs for the next, forming a robust data-processing pipeline.

**Data Flow Diagram:**

```
[Master CSV] -> [analysis_generator.py] -> [quantitative.json] + [qualitative.json]
                                                  |                     |
                                                  V                     V
[Master CSV] + [quantitative.json] -> [graph_generator.py] -> [output/graphs/]
                                                                     |
                                                                     V
[quantitative.json] + [qualitative.json] + [output/graphs/] -> [document_assembler.py] -> [Final_Report.docx]
```

---

### **File 1: `analysis_generator.py` - The Analyst System**

**Core Purpose:** To perform all data analysis (quantitative and qualitative) and distill the results into structured, machine-readable JSON files. This script is the "brain" of the operation.

**Agent-Level Breakdown:**

**1. `ConfigurationAgent`**
*   **Role:** The System Initializer.
*   **Purpose:** To validate the environment and data schema before processing begins.
*   **Inputs:** `MASTER_CSV_PATH` (constant).
*   **Key Functions:**
    *   `validate_master_csv_schema()`: Checks if the master CSV contains all required columns for the analysis (e.g., `Code Language`, `Industry`, and the various `{Model} Human {Dimension}` columns). This is where you would ensure columns like `Chatgpt Human Correctness` are found.
    *   `determine_optimal_workers()`: Sets the concurrency level for the pipeline.
*   **Output:** A validated configuration state, allowing the pipeline to proceed.

**2. `DataSlicerTool`**
*   **Role:** The Data Architect.
*   **Purpose:** To slice the master data into logical units of work and pre-calculate all quantitative metrics.
*   **Inputs:** The master DataFrame loaded from the CSV.
*   **Key Functions:**
    *   `compute_quantitative_analysis(df)`: This is a **critical function**. It will group the DataFrame by `Code Language` and `Industry`. For each group, it calculates the score distributions (counts of 5s, 4s, etc.) and determines the winner for each dimension (`Relevance`, `Completeness`, `Correctness`).
    *   `create_analysis_states(master_df)`: Iterates through each `Language-Domain-Dimension` combination. For each one, it creates a small, filtered `data_slice` DataFrame and bundles it with its corresponding pre-computed quantitative context into an `AnalysisState` object.
*   **Output:**
    1.  A list of `AnalysisState` objects, ready for parallel processing.
    2.  The complete, aggregated data from `compute_quantitative_analysis`, which will be the source for `quantitative_results.json`.

**3. `AnalysisAgent`**
*   **Role:** The AI Qualitative Analyst.
*   **Purpose:** To generate the detailed, human-like textual analysis for a single unit of work.
*   **Inputs:** A single `AnalysisState` object.
*   **Key Functions:**
    *   `generate_analysis(state)`: Takes the `state`, which includes the data slice and the quantitative context.
    *   `_create_enhanced_prompt(state, csv_data)`: Constructs a sophisticated prompt for the LLM, intelligently weaving in the EDA report and quantitative context (e.g., "The winner was Model X with Y fives...") to guide the LLM's response.
*   **Output:** The `AnalysisState` object, now enriched with the generated `analysis_report` string.

**4. `ReportAggregator` (Modified for this Architecture)**
*   **Role:** The Data Contract Publisher.
*   **Purpose:** To collect all processed results and save them into the structured JSON files that serve as the contract for the other scripts.
*   **Inputs:** A list of all completed `AnalysisState` objects.
*   **Key Functions:**
    *   `aggregate_results(completed_states)`: Main orchestration method.
    *   **`save_quantitative_json()`**: Collects the `quantitative_context` from all states and aggregates it into a single, clean JSON file. This file will be the "single source of truth" for all numerical data.
        *   **Output (`quantitative_results.json`):**
            ```json
            {
              "FinTech (Python)": {
                "Relevance": { "Chatgpt": {"score_5": 10, ...}, "Gemini": {...} },
                "Completeness": { "Chatgpt": {"score_5": 8, ...}, "Claude": {...} }
              },
              ...
            }
            ```
    *   **`save_qualitative_json()`**: Iterates through the completed states and extracts the generated text. It creates a predictable `analysis_id` for each entry to serve as a universal key.
        *   **Output (`qualitative_analysis.json`):**
            ```json
            [
              {
                "analysis_id": "Python-FinTech-Completeness",
                "language": "Python",
                "domain": "FinTech",
                "dimension": "Completeness",
                "report_text": "The winner is Claude Opus 4..."
              },
              {
                "analysis_id": "Python-Machine_Learning-Relevance",
                "language": "Python",
                "domain": "Machine Learning",
                "dimension": "Relevance",
                "report_text": "The winner is Gemini 2.5 pro..."
              }
            ]
            ```

---

### **File 2: `graph_generator.py` - The Agentic Visualizer**

**Core Purpose:** To read the structured data produced by the Analyst System and generate all required image files, saving them with predictable, key-based filenames.

**Agent-Level Breakdown:**

**1. `GraphOrchestratorAgent`**
*   **Role:** The Visualization Pipeline Manager.
*   **Purpose:** To manage the end-to-end process of graph creation.
*   **Inputs:** Paths to `quantitative_results.json` and the master CSV.
*   **Key Functions:**
    *   `run_full_generation()`: Loads the necessary data files and calls the specialist agents in order.
*   **Output:** A console report of the generation process (e.g., "Generated 50 drill-down charts and 4 summary grids.").

**2. `DrillDownGrapherAgent`**
*   **Role:** The Detailed Chart Specialist.
*   **Purpose:** To create the specific bar charts that compare models for a single `Language-Domain-Dimension`.
*   **Inputs:** The data from `quantitative_results.json`.
*   **Key Functions:**
    *   `generate_all_charts(data)`: Iterates through the quantitative data. For each `Language-Domain-Dimension` entry, it calls the single chart generation function. This can be parallelized.
    *   `_generate_single_chart(chart_data, language, domain, dimension)`: This is the core plotting function. It uses `matplotlib`/`seaborn` to create a bar chart of the '5' scores for each model.
    *   **The Linking Mechanism:** It saves the chart using the universal key: `output/graphs/drilldown/{language}-{domain}-{dimension}.png`. This filename predictability is critical for the next stage.

**3. `SummaryGridGrapherAgent`**
*   **Role:** The High-Level View Artist.
*   **Purpose:** To create the 4x4 summary grids that show a metric across all industries for a given language.
*   **Inputs:** The master DataFrame.
*   **Key Functions:**
    *   `generate_all_grids(df)`: Main public method.
    *   `_generate_single_grid(df_filtered, language, dimension)`: Creates a `matplotlib` figure with a grid of subplots. It then iterates through each `Industry` in the filtered DataFrame, plotting a small bar chart onto the appropriate subplot axis.
    *   **The Linking Mechanism:** It saves the composite image with a predictable name: `output/graphs/summary/{language}-{dimension}_summary.png`.

---

### **File 3: `document_assembler.py` - The Agentic Publisher**

**Core Purpose:** To act as the final assembly line, intelligently combining the text, tables, and images from the previous stages into a polished, professional `.docx` report.

**Agent-Level Breakdown:**

**1. `DocumentAssemblyAgent` (The Conductor)**
*   **Role:** The Report Orchestrator.
*   **Purpose:** Manages the entire document creation workflow.
*   **Inputs:** Paths to the `qualitative.json`, `quantitative.json`, and the `output/graphs` directory.
*   **Key Functions:**
    *   `create_final_report()`: Initializes a `python-docx` Document object, loads the JSON data, and calls the writer agents sequentially, passing the document object to each one.
*   **Output:** The final `Final_Report.docx` file.

**2. `StaticContentWriterAgent`**
*   **Role:** The Scribe.
*   **Purpose:** To write the non-dynamic, boilerplate sections of the report.
*   **Inputs:** The `docx` Document object.
*   **Key Functions:** `add_title_page()`, `add_table_of_contents()`, `add_abstract()`.
*   **Output:** The `docx` Document object, populated with introductory content.

**3. `TabularDataWriterAgent`**
*   **Role:** The Table Master.
*   **Purpose:** To create structured summary tables from the quantitative data.
*   **Inputs:** The `docx` Document object and the data from `quantitative_results.json`.
*   **Key Functions:** `create_summary_score_table(data)`: Programmatically builds a table in the document, populating headers and rows with the aggregated scores.
*   **Output:** The `docx` Document object, now containing summary tables.

**4. `QualitativeAnalysisWriterAgent`**
*   **Role:** The Intelligent Assembler.
*   **Purpose:** To perform the most complex task: weaving the individual analyses, detailed graphs, and summary grids into a coherent narrative.
*   **Inputs:** The `docx` Document object, data from `qualitative.json`, and the file paths to the graphs.
*   **Key Functions:**
    *   `write_analysis_sections(data)`: The main method. It will group the `qualitative.json` data by `dimension`.
    *   **Intelligence and Linking Logic (Inside a loop for each analysis object):**
        1.  When starting a new dimension (e.g., "Relevance"), it first constructs the summary grid path (e.g., `output/graphs/summary/Python-Relevance_summary.png`) and inserts that image.
        2.  It then iterates through each individual analysis object from the JSON.
        3.  It reads the `analysis_id` (e.g., "Python-FinTech-Completeness").
        4.  It writes the textual heading (e.g., "4.2.2 FinTech (Python)").
        5.  **It constructs the drill-down graph's file path:** `f"output/graphs/drilldown/Python-FinTech-Completeness.png"`.
        6.  It calls `document.add_picture()` with this known path.
        7.  It takes the `report_text` from the JSON object and writes it into the document, applying formatting as needed.
*   **Output:** The `docx` Document object, now containing the fully detailed, illustrated analysis sections.