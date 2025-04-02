import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from groq import Groq

class InferencePipeline:
    def __init__(self, dataset_path, groq_api_key):
        self.batch_size = 5
        self.dataset = self.load_dataset(dataset_path)
        self.groq_client = Groq(api_key=groq_api_key)

        self.segment_mapping = {
            'Food/Beverage': '50000000',
            'Healthcare': '51000000',
            'Home Appliances': '72000000',
            'Electrical Supplies': '78000000',
            'Clothing': '67000000',
            "Computing":"65000000",
            "Tools/Equipment":"80000000",
            "Camping":"74000000",
            "Arts/Crafts/Needlework":"70000000",
            "Music":"61000000",

        }

        self.family_mapping = {
    'Bread/Bakery Products': '50180000',
    'Beverages': '50200000',
    'Vegetables - Unprepared/Unprocessed (Frozen)': '50290000',
    'Fruits/Vegetables/Nuts/Seeds Prepared/Processed': '50100000',
    'Fruits - Unprepared/Unprocessed (Fresh)':'50250000',
    'Fruits/Vegetables Fresh Cut': '50380000',
    'Fruits/Vegetables Fresh & Fresh Cut' :'50370000',
    'Fruits - Unprepared/Unprocessed (Frozen)' :'50270000',
    'Fruits - Unprepared/Unprocessed (Shelf Stable)': '50310000',
    'Cereal/Grain/Pulse Products': '50220000',
    'Prepared/Preserved Foods': '50190000',
    'Meat/Poultry/Other Animals': '50240000',
    'Nuts/Seeds - Unprepared/Unprocessed (Perishable)': '50330000',
    'Nuts/Seeds - Unprepared/Unprocessed (In Shell)': '50340000',
    'Vegetables (Non Leaf) - Unprepared/Unprocessed (Fresh)': '50260000',
    'Vegetables - Unprepared/Unprocessed (Shelf Stable)' :'50320000',
    'Leaf Vegetables - Unprepared/Unprocessed (Fresh)' :'50350000',
    'Fish and Seafood': '50120000',
    'Food/Beverage Variety Packs': '50230000',
    'Meat/Fish/Seafood Substitutes': '50390000',
    'Milk/Butter/Cream/Yogurts/Cheese/Eggs/Substitutes': '50130000',
    'Seasonings/Preservatives/Extracts': '50170000',
    'Oils/Fats Edible': '50150000',
    'Confectionery/Sugar Sweetening Products': '50160000',
    'Health Treatments/Aids': '51100000',
    'Pharmaceutical Drugs': '51160000',
    'Health Enhancement': '51120000',
    'Healthcare Variety Packs': '51140000',
    'Medical Devices': '51150000',
    'Home Diagnostics': '51130000',
    'Veterinary Healthcare': '51170000',
    'Family Planning': '51110000',
    'Small Domestic Appliances': '72020000',
    'Major Domestic Appliances': '72010000',
    'Activewear': '67030000',
    'Clothing': '67010000',
    'Protective Wear': '67050000',
    'Swimwear': '67060000',
    'Sleepwear': '67020000',
    'Underwear': '67040000',
    'General Electrical Hardware': '78060000',
    'Electronic Communication Components': '78050000',
    'Electrical Lighting': '78030000',
    'Electrical Cabling/Wiring': '78040000',
    'Electrical Connection/Distribution': '78020000',
    'Computers/Video Games': '65010000',
    'Tools/Equipment': '74010000',
    'Camping': '75010000',
    'Arts/Crafts/Needlework Supplies': '70010000',
    'Musical Instruments/Accessories': '61010000'
}

        self.candidate_labels = {
            'segments': list(self.segment_mapping.keys()),
            'families': list(self.family_mapping.keys()),
        }

        self.segment_to_families = {
    "Food/Beverage": [
        'Fruits/Vegetables/Nuts/Seeds Prepared/Processed',
        'Fish and Seafood',
        'Milk/Butter/Cream/Yogurts/Cheese/Eggs/Substitutes',
        'Oils/Fats Edible',
        'Confectionery/Sugar Sweetening Products',
        'Seasonings/Preservatives/Extracts',
        'Bread/Bakery Products',
        'Prepared/Preserved Foods',
        'Beverages',
        'Cereal/Grain/Pulse Products',
        'Meat/Poultry/Other Animals',
        'Vegetables - Unprepared/Unprocessed (Frozen)',
        'Fruits - Unprepared/Unprocessed (Frozen)',
        'Fruits - Unprepared/Unprocessed (Fresh)',
        'Fruits/Vegetables Fresh Cut',
        'Food/Beverage Variety Packs',
        'Meat/Fish/Seafood Substitutes',
        'Vegetables - Unprepared/Unprocessed (Shelf Stable)',
        'Fruits - Unprepared/Unprocessed (Shelf Stable)',
        'Nuts/Seeds - Unprepared/Unprocessed (Perishable)',
        'Nuts/Seeds - Unprepared/Unprocessed (In Shell)',
        'Fruits/Vegetables Fresh & Fresh Cut',
        'Vegetables (Non Leaf) - Unprepared/Unprocessed (Fresh)',
        'Leaf Vegetables - Unprepared/Unprocessed (Fresh)'
    ],
    "Healthcare": [
        'Family Planning',
        'Health Enhancement',
        'Healthcare Variety Packs',
        'Medical Devices',
        'Pharmaceutical Drugs',
        'Health Treatments/Aids',
        'Veterinary Healthcare',
        'Home Diagnostics'
    ],
    "Clothing": [
        'Clothing',
        'Activewear',
        'Underwear',
        'Swimwear',
        'Protective Wear',
        'Sleepwear'
    ],
    "Home Appliances": [
        'Major Domestic Appliances',
        'Small Domestic Appliances'
    ],
    "Electrical Supplies": [
        'Electrical Connection/Distribution',
        'Electrical Lighting',
        'Electrical Cabling/Wiring',
        'General Electrical Hardware'
        'Electronic Communication Components'
    ],
    "Computing": [
        'Computers/Video Games'
    ],
    "Tools/Equipment": [
        'Tools/Equipment'
    ],
    "Camping": [
        'Camping'
    ],
    "Arts/Crafts/Needlework": [
        'Arts/Crafts/Needlework Supplies'
    ],
    "Music": [
        'Musical Instruments/Accessories'
    ]
}

    def load_dataset(self, path):
        """Load dataset and validate required columns."""
        required_columns = ['item_name', 'item_segment', 'item_family']
        df = pd.read_csv(path)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df['item_segment'] = df['item_segment'].fillna('Unknown').astype('category')
        df['item_family'] = df['item_family'].fillna('Unknown').astype('category')

        return df

    def process_batch(self, batch, model_name):
        """Process a batch of items with two-step inference:
           1. Classify the segment.
           2. Classify the family using only families for that segment.
        """
        responses = []
        start_time = time.time()

        for item in batch:
            try:
                prompt_seg = (
                    f"Classify the product: {item['item_name']}\n"
                    f"Options - Segments: {self.candidate_labels['segments']}\n"
                    "Respond with JSON containing:\n"
                    "- segment_name (string)"
                )
                seg_response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_seg}],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                seg_data = json.loads(seg_response.choices[0].message.content)
                predicted_segment = seg_data.get('segment_name', '').strip()


                families_for_segment = self.segment_to_families.get(predicted_segment, self.candidate_labels['families'])
                prompt_family = (
                    f"Classify the product: {item['item_name']}\n"
                    f"Options - Families under the segment '{predicted_segment}': {families_for_segment}\n"
                    "Respond with JSON containing:\n"
                    "- family_name (string)"
                )
                fam_response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_family}],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                fam_data = json.loads(fam_response.choices[0].message.content)
                predicted_family = fam_data.get('family_name', '').strip()

                responses.append({
                    "item_name": item['item_name'],
                    "segment_code": self.segment_mapping.get(predicted_segment, ''),
                    "segment_name": predicted_segment,
                    "family_code": self.family_mapping.get(predicted_family, ''),
                    "family_name": predicted_family
                })
            except Exception as e:
                print(f"Error processing {item['item_name']}: {str(e)}")
                responses.append({
                    "item_name": item['item_name'],
                    "segment_code": '',
                    "segment_name": '',
                    "family_code": '',
                    "family_name": ''
                })

        inference_time = time.time() - start_time
        return responses, inference_time / len(batch)

    def evaluate_model(self, model_name, sample):
        """Evaluate the model using micro F1-score, accuracy, and average inference time."""
        true_segments = sample['item_segment'].cat.codes.tolist()
        true_families = sample['item_family'].cat.codes.tolist()

        preds_segment = []
        preds_family = []
        inference_times = []

        for i in range(0, len(sample), self.batch_size):
            batch = sample.iloc[i:i + self.batch_size].to_dict('records')
            batch_preds, batch_time = self.process_batch(batch, model_name)

            preds_segment.extend([
                sample['item_segment'].cat.categories.get_loc(p['segment_name'])
                if p['segment_name'] in sample['item_segment'].cat.categories else -1
                for p in batch_preds
            ])

            preds_family.extend([
                sample['item_family'].cat.categories.get_loc(p['family_name'])
                if p['family_name'] in sample['item_family'].cat.categories else -1
                for p in batch_preds
            ])

            inference_times.append(batch_time)

        return {
            'accuracy_segment': accuracy_score(true_segments, preds_segment),
            'f1_segment': f1_score(true_segments, preds_segment, average='micro'),
            'accuracy_family': accuracy_score(true_families, preds_family),
            'f1_family': f1_score(true_families, preds_family, average='micro'),
            'avg_inference_time': np.mean(inference_times)
        }

    def run_pipeline(self, models, output_dir):
        """Run the pipeline: generate predictions and evaluate models."""
        os.makedirs(output_dir, exist_ok=True)

        eval_sample = self.dataset.sample(frac=0.2, random_state=42)
        results = {}

        for model in models:
            all_predictions = []

            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset.iloc[i:i + self.batch_size].to_dict('records')
                batch_preds, batch_time = self.process_batch(batch, model)

                all_predictions.extend([{
                    'Item_name': item['item_name'],
                    'true_label': f"{item['item_segment']}|{item['item_family']}",
                    'prediction': f"{pred['segment_name']}|{pred['family_name']}",
                    'inference_time': batch_time
                } for item, pred in zip(batch, batch_preds)])

            gpu_type = 'GROQ-LPU'
            filename = f"{output_dir}/{model.replace('/', '-')}_{gpu_type}.csv"
            pd.DataFrame(all_predictions).to_csv(filename, index=False)
            results[model] = self.evaluate_model(model, eval_sample)

        with open(f"{output_dir}/performance.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == '__main__':
    config = {
        'dataset_path': '/content/sampled_family_df_english.csv',
        'groq_api_key': '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',
        'models_to_test': [
            'gemma2-9b-it'
        ],
        'output_dir': './english_results2'
    }

    pipeline = InferencePipeline(config['dataset_path'], config['groq_api_key'])
    performance = pipeline.run_pipeline(config['models_to_test'], config['output_dir'])

    print("\nPerformance Results:", json.dumps(performance, indent=2))
