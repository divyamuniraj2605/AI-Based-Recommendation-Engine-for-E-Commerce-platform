import pandas as pd
from preprocess_data import process_data

raw_data = pd.read_csv("Clean_data.csv")
data = process_data(raw_data)

from content_based_filtering import content_based_recommendation

def evaluate_content_based_metrics(data, item_name,top_n=10):
    print(f"\n--- FULL METRICS: CONTENT-BASED (Top {top_n})---")

    #Check the item exists
    item_matches = data[data['Name']==item_name]
    if item_matches.empty:
        print(f"item {item_name} Not Found")
        return True
    
    item_data = item_matches.iloc[0]
    item_category = item_data.get('Category',None)
    item_brand = item_data.get('Brand','Unknown')

    print(f"Input: '{item_name}'")
    print(f"Category: {item_category}, Brand: {item_brand}")

    #2. All Relevant items in the dataset
    relevant_items = set()
    if item_category:
        relevant_items.update(data[data['Category']==item_category]['Name'].values)
    relevant_items.update(data[data['Brand']==item_category]['Name'].values)
    total_relevant = len(relevant_items) - 1 
    print(f"Total relevant items in dataset: {total_relevant}")

    #3.Get Recommendation
    recs = content_based_recommendation(data,item_name,top_n)
    if recs.empty:
        print("No Recommendations!")
        return None
    
    recommend_names = set(recs['Name'].values)

    #4.Calculate All Metrics
    true_positives = len(recommend_names & relevant_items)

    precision = true_positives / top_n
    recall = true_positives / total_relevant if total_relevant >0 else 0
    f1 = 2 * (precision * recall) / (precision * recall) if (precision + recall) > 0 else 0

    #5.Results
    print(f"\n Metrics-{top_n}")
    print(f"Precision: {precision:.3f} ({true_positives}/{top_n})")
    print(f" Recall: {recall:.3f} ({true_positives}/{total_relevant})")
    print(f"F1-Score: {f1:.3f}")

    print(f"\n Matches: {list(recommend_names & relevant_items)[:3]}...")
    return {f'precision': precision, 'recall':recall, 'f1':f1}

#Main block
if __name__ == "__main__":
    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)

    item_name = data['Name'].iloc[0]
    evaluate_content_based_metrics(data, item_name, top_n=10)