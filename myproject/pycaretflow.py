from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment
import json

# 1. ดึงข้อมูล (ประกาศตัวแปร data ให้ชัดเจน)
data = get_data('diabetes') 

# 2. Setup ML Pipeline
exp = ClassificationExperiment()
exp.setup(data, target='Class variable', session_id=123, verbose=False)
# 3. เปรียบเทียบโมเดล และเลือกตัวที่ดีที่สุด
best_model = exp.compare_models()

# 4. ดึง Performance Metadata ออกมา (ฝังลงใน JSON)
results = exp.pull() # ดึงตารางผลลัพธ์ล่าสุด
top_model_stats = results.iloc[0].to_dict()

metadata = {
    "model_name": str(best_model).split('(')[0],
    "accuracy": top_model_stats['Accuracy'],
    "auc": top_model_stats['AUC'],
    "f1": top_model_stats['F1'],
    "recommendation": "Ready for Deployment" if top_model_stats['Accuracy'] > 0.75 else "Needs Retraining"
}

with open('model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

print("\n--- ML Pipeline Completed: Metadata Saved to model_metadata.json ---")

# 5. Simple Agent AI สำหรับสรุปผล
def summary_agent(metadata_path):
    with open(metadata_path, 'r') as f:
        m = json.load(f)
    
    summary = f"""
    🤖 [AI Agent Report]
    จากการประเมินโมเดล {m['model_name']} 
    พบว่ามีค่า Accuracy อยู่ที่ {m['accuracy']:.2%} และ F1-Score {m['f1']:.2%}
    สถานะผลลัพธ์: {m['recommendation']}
    """
    return summary

print(summary_agent('model_metadata.json'))