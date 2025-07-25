
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, inspect
import spacy
from collections import defaultdict
import os
import subprocess
import time
import nbformat
from nbconvert import PythonExporter
import tempfile


try:
    nlp = spacy.load("en_core_web_md") 
except OSError:
    st.warning("Downloading spaCy model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")




engine = create_engine("sqlite:///reddit_sentiment.db")


def get_table_name(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "reddit_posts" in tables:
        return "reddit_posts"
    elif "reddit_sentiment" in tables:
        return "reddit_sentiment"
    return None

def extract_key_phrases(text, brand):
    doc = nlp(text.lower())
    phrases = set()
    

    for chunk in doc.noun_chunks:
        if brand.lower() in chunk.text:
            phrases.add(chunk.text.replace(brand.lower(), "").strip())
        elif any(t.text.lower() in ["great","poor","love","hate","excellent","bad"] for t in chunk.root.head.children):
            phrases.add(chunk.text)
    
    return [p for p in phrases if len(p) > 3 and p not in {"it","this","that"}]

def run_notebook_analysis(brand_name):
    """Execute the notebook to analyze a new brand"""
    with st.status(f"🔍 Analyzing new brand: {brand_name}...", expanded=True) as status:
        try:

            notebook_path = "Untitled-1.ipynb"
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            exporter = PythonExporter()
            python_script, _ = exporter.from_notebook_node(nb)
            

            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                encoding='utf-8', 
                delete=False
            ) as tmp:
                modified_script = python_script.replace(
                    'brand = input("Enter brand name (e.g., Tesla, Nike): ").strip()',
                    f'brand = "{brand_name}"'
                )
                tmp.write(modified_script)
                tmp_path = tmp.name
            
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                status.update(label=f"Successfully analyzed {brand_name}!", state="complete")
                return True
            else:
                st.error(f"Analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            return False
        finally:

            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

def main():
    st.set_page_config(layout="wide", page_title="Brand Sentiment Overview")
    

    if not os.path.exists("reddit_sentiment.db"):
        st.error("Database not found. Please run your notebook first.")
        return
    
    table_name = get_table_name(engine)
    if table_name is None:
        st.error("No valid tables found in database")
        return

    st.sidebar.title("Analysis Mode")
    analysis_mode = st.sidebar.radio(
        "Select analysis type:",
        ["Existing Brand", "New Brand"],
        index=0
    )
    
    if analysis_mode == "Existing Brand":
        brands = pd.read_sql(f"SELECT DISTINCT brand FROM {table_name}", engine)['brand'].tolist()
        if not brands:
            st.warning("No brands found in database")
            return
            
        selected_brand = st.selectbox("Select Brand", brands)
        show_analysis = True
    else:
        new_brand = st.text_input("Enter new brand name to analyze")
        if new_brand:
            if st.button("Run Analysis"):
                if run_notebook_analysis(new_brand):
                    st.success(f"Successfully analyzed {new_brand}!")
                    st.rerun()  
                else:
                    st.error("Failed to analyze brand")
            show_analysis = False
        else:
            st.info("Please enter a brand name")
            show_analysis = False
    
    if show_analysis:

        st.header(f" {selected_brand} Sentiment Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total = pd.read_sql(f"SELECT COUNT(*) FROM {table_name} WHERE brand='{selected_brand}'", engine).iloc[0,0]
            st.metric("Total Analyzed", total)
        
        with col2:
            positive = pd.read_sql(f"SELECT COUNT(*) FROM {table_name} WHERE brand='{selected_brand}' AND sentiment='positive'", engine).iloc[0,0]
            st.metric("Positive", positive, f"{positive/total:.1%}")
        
        with col3:
            negative = pd.read_sql(f"SELECT COUNT(*) FROM {table_name} WHERE brand='{selected_brand}' AND sentiment='negative'", engine).iloc[0,0]
            st.metric("Negative", negative, f"{negative/total:.1%}")
        
        with col4:
            neutral = pd.read_sql(f"SELECT COUNT(*) FROM {table_name} WHERE brand='{selected_brand}' AND sentiment='neutral'", engine).iloc[0,0]
            st.metric("Neutral", neutral, f"{neutral/total:.1%}")
        
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            pd.read_sql(f"SELECT sentiment, COUNT(*) as count FROM {table_name} WHERE brand='{selected_brand}' GROUP BY sentiment", engine),
            names="sentiment",
            values="count",
            color="sentiment",
            color_discrete_map={"positive":"green","negative":"red","neutral":"gray"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("🔍 Key Features Driving Sentiment")
        
        samples = pd.read_sql(f"""
            SELECT text, sentiment 
            FROM {table_name} 
            WHERE brand='{selected_brand}' 
            ORDER BY RANDOM() 
            LIMIT 100
        """, engine)
        
        features = defaultdict(lambda: {"positive":0, "negative":0, "neutral":0})
        
        for _, row in samples.iterrows():
            for phrase in extract_key_phrases(row['text'], selected_brand):
                features[phrase][row['sentiment']] += 1
        
        features_df = pd.DataFrame([
            {"feature":f, **counts} 
            for f, counts in features.items() 
            if sum(counts.values()) >= 3
        ])
        
        if not features_df.empty:
            features_df["total"] = features_df.sum(axis=1)
            
            st.subheader("Top Positive Drivers")
            pos_features = features_df.nlargest(5, "positive")[["feature","positive"]]
            st.dataframe(pos_features.style.background_gradient(cmap="Greens"), hide_index=True)
            
            st.subheader("Top Negative Drivers")
            neg_features = features_df.nlargest(5, "negative")[["feature","negative"]]
            st.dataframe(neg_features.style.background_gradient(cmap="Reds"), hide_index=True)
            
            st.subheader("Example Mentions")
            sentiment_type = st.radio("Show examples for:", ["positive","negative"], key="sentiment_type", horizontal=True)
            
            examples = pd.read_sql(f"""
                SELECT text 
                FROM {table_name} 
                WHERE brand='{selected_brand}' 
                AND sentiment='{sentiment_type}'
                ORDER BY upvotes DESC 
                LIMIT 3
            """, engine)
            
            for example in examples['text']:
                st.markdown(f"- {example[:200]}..." if len(example) > 200 else f"- {example}")
        else:
            st.warning("No significant features detected. Try analyzing more data.")

if __name__ == "__main__":
    main()