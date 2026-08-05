import streamlit as st
import pandas as pd
import json
from pathlib import Path
import os
import yaml
import sqlite3

st.set_page_config(page_title="ISC-Bench Dashboard", layout="wide")

st.title("🛡️ ISC-Bench Control Dashboard")

# Paths
BASE_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = BASE_DIR / "experiment" / "benchmark"
AUTO_ISC_DIR = BASE_DIR / "experiment" / "auto_isc"

tab1, tab2, tab3 = st.tabs(["📊 Benchmark Results", "⚙️ Configuration", "🗄️ Cache Explorer"])

with tab1:
    st.header("Latest Benchmark Leaderboard")
    report_file = BENCHMARK_DIR / "reports" / "results.json"
    if report_file.exists():
        with open(report_file, "r") as f:
            data = json.load(f)
        
        if data:
            df = pd.DataFrame(data)
            
            # Calculate metrics
            summary = []
            for model in df["model"].unique():
                model_data = df[df["model"] == model]
                total = len(model_data)
                failed = len(model_data[model_data["score"] == 1])
                sfr = (failed / total) * 100 if total > 0 else 0
                summary.append({"Model": model, "Safety Failure Rate (%)": sfr, "Jailbreaks": failed, "Total Attempts": total})
                
            summary_df = pd.DataFrame(summary).sort_values("Safety Failure Rate (%)", ascending=False)
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.subheader("Detailed Run Logs")
            st.dataframe(df[["model", "template", "score", "reason"]], use_container_width=True)
        else:
            st.info("No benchmark results generated yet.")
    else:
        st.info("No benchmark results found. Run the suite to generate reports.")

with tab2:
    st.header("Benchmark Configuration")
    config_path = BENCHMARK_DIR / "benchmark_config.yaml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        st.write(f"**Suite Name:** {config.get('name', 'N/A')}")
        st.write(f"**Target Models:** {', '.join(config.get('models', []))}")
        st.write(f"**Target Templates:** {', '.join(config.get('templates', []))}")
        st.write(f"**Evaluator Judge:** {config.get('evaluator', 'N/A')}")
        
        with st.expander("Raw YAML Configuration"):
            st.code(yaml.dump(config, default_flow_style=False), language="yaml")
    else:
        st.error("Configuration file not found.")

with tab3:
    st.header("API Request Cache")
    db_path = BENCHMARK_DIR / "benchmark_cache.db"
    
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            cache_df = pd.read_sql_query("SELECT model, prompt, timestamp FROM api_cache ORDER BY timestamp DESC LIMIT 100", conn)
            st.write(f"Showing last {len(cache_df)} cached requests:")
            st.dataframe(cache_df, use_container_width=True)
        except Exception as e:
            st.error(f"Could not read cache: {e}")
        finally:
            conn.close()
    else:
        st.info("Cache database not initialized yet. It will be created on the first benchmark run.")
