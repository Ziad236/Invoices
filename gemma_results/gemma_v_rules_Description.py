import os
import json
import time
import pandas as pd
import numpy as np
from groq import Groq
from sklearn.metrics import accuracy_score, f1_score

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
        }

        self.family_mapping = {
            'Bread/Bakery Products': '50180000',
            'Beverages': '50200000',
            'Cereal/Grain/Pulse Products': '50220000',
            'Prepared/Preserved Foods': '50190000',
            'Food/Beverage Variety Packs': '50230000',
            'Seasonings/Preservatives/Extracts': '50170000',
            'Oils/Fats Edible': '50150000',
            'Confectionery/Sugar Sweetening Products': '50160000',
            'Health Treatments/Aids': '51100000',
            'Health Enhancement': '51120000',
            'Family Planning': '51110000',
            'Small Domestic Appliances': '72020000',
            'Major Domestic Appliances': '72010000',
            'Activewear': '67030000',
            'Clothing': '67010000',
            'Protective Wear': '67050000',
            'Swimwear': '67060000',
            'Underwear': '67040000',
            'General Electrical Hardware': '78060000',
            'Electrical Lighting': '78030000',
            'Electrical Cabling/Wiring': '78040000',
            'Electrical Connection/Distribution': '78020000',
        }

        self.candidate_labels = {
            'segments': list(self.segment_mapping.keys()),
            'families': list(self.family_mapping.keys())
        }

        self.examples = {
            'Food/Beverage': {
                'Bread/Bakery Products': ['Pies','Pizzas','Cake', 'Bread', 'Cookies','Biscuits','Baking Mix','Cooking Mix'],
                'Beverages': ['Orange Juice', 'Coffee', 'Tea'],
                'Cereal/Grain/Pulse Products': ['Oats', 'Rice', 'Lentils'],
                'Prepared/Preserved Foods': ['Canned Soup', 'Frozen Pizza', 'Ready Meals'],
                'Food/Beverage Variety Packs': ['Snack Box', 'Breakfast Pack', 'Picnic Set'],
                'Seasonings/Preservatives/Extracts': ['Salt', 'Vinegar', 'Vanilla Extract'],
                'Oils/Fats Edible': ['Olive Oil', 'Butter', 'Coconut Oil'],
                'Confectionery/Sugar Sweetening Products': ['Chocolate', 'Candy', 'Honey']
            },
            'Healthcare': {
                'Health Treatments/Aids': ['Bandages', 'Thermometer', 'First Aid Kit'],
                'Health Enhancement': ['Vitamins', 'Protein Powder', 'Energy Bars'],
                'Family Planning': ['Condoms', 'Pregnancy Tests', 'Contraceptive Pills']
            },
            'Home Appliances': {
                'Small Domestic Appliances': ['Blender', 'Coffee Maker', 'Microwave'],
                'Major Domestic Appliances': ['Refrigerator', 'Washing Machine' ,'Warmer','Burner']
            },
            'Clothing': {
                'Activewear': ['Running Shoes', 'Yoga Pants', 'Sports Bra'],
                'Clothing': ['T-Shirt', 'Jeans', 'Jacket'],
                'Protective Wear': ['Helmet', 'Gloves', 'Safety Boots'],
                'Swimwear': ['Swimsuit', 'Swim Trunks', 'Bikini'],
                'Underwear': ['Socks', 'Boxers', 'Bras']
            },
            'Electrical Supplies': {
                'General Electrical Hardware': ['Screwdrivers', 'Pliers', 'Wrenches', 'Voltage Tester', 'Wire Stripper'],
                'Electrical Lighting': ['LED Bulbs', 'Lamps', 'Chandeliers', 'Flashlights', 'Floodlights'],
                'Electrical Cabling/Wiring': ['Extension Cords', 'Ethernet Cables', 'Power Strips', 'Coaxial Cables', 'HDMI Cables'],
                'Electrical Connection/Distribution': ['Circuit Breakers', 'Fuse Boxes', 'Switchboards', 'Electrical Panels', 'Distribution Boards']
            }
        }

    def load_dataset(self, path):
        """Load and validate dataset"""
        required_columns = ['item_name', 'item_segment', 'item_family', 'item_price']
        df = pd.read_csv(path)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df['item_segment'] = df['item_segment'].fillna('Unknown').astype('category')
        df['item_family'] = df['item_family'].fillna('nan').astype('category')
        df['item_price'] = df['item_price'].str.replace(',', '').fillna(0).astype(float)

        return df

    def process_batch(self, model_name, batch):
        start_time = time.time()
        results = self._process_groq(batch, model_name)
        inference_time = time.time() - start_time
        return results, inference_time

    def _process_groq(self, batch, model_name):
        """Process Groq API requests with rate limiting"""
        responses = []
        for item in batch:
            try:
                segment = item['item_segment']
                family = item['item_family']
                examples = self.examples.get(segment, {}).get(family, [])

                prompt = f"""Classify product: {item['item_name']} (Price: {item['item_price']} SAR)
                             Options - Segments: {self.candidate_labels['segments']}
                             Families: {self.candidate_labels['families']}
                             Examples for {segment} - {family}: {', '.join(examples)}
                             Family Descriptions:
                             - Bread/Bakery Products: Includes items like bread, cakes, cookies, and other baked goods.
                             - Beverages: Includes drinks such as juices, coffee, tea, and soft drinks.
                             - Cereal/Grain/Pulse Products: Includes grains, cereals, lentils, and other pulse products.
                             - Prepared/Preserved Foods: Includes canned, frozen, and ready-to-eat meals.
                             - Food/Beverage Variety Packs: Includes assorted food and beverage packs.
                             - Seasonings/Preservatives/Extracts: Includes spices, herbs, preservatives, and flavor extracts.
                             - Oils/Fats Edible: Includes cooking oils, butter, and other edible fats.
                             - Confectionery/Sugar Sweetening Products: Includes chocolates, candies, and sweeteners.
                             - Health Treatments/Aids: Includes medical aids like bandages, thermometers, and first aid kits.
                             - Health Enhancement: Includes vitamins, supplements, and health boosters.
                             - Family Planning: Includes contraceptives, pregnancy tests, and family planning products.
                             - Small Domestic Appliances: Includes small kitchen and home appliances like blenders, coffee makers, and microwaves.
                             - Major Domestic Appliances: Includes large appliances like refrigerators, washing machines, and ovens.
                             - Activewear: Includes sportswear like running shoes, yoga pants, and sports bras.
                             - Clothing: Includes everyday wear like t-shirts, jeans, and jackets.
                             - Protective Wear: Includes safety gear like helmets, gloves, and safety boots.
                             - Swimwear: Includes swimsuits, swim trunks, and bikini.
                             - Underwear: Includes undergarments like socks, boxers, and bras.
                             - General Electrical Hardware: Includes tools like screwdrivers, pliers, and wrenches.
                             - Electrical Lighting: Includes lighting products like bulbs, lamps, and flashlights.
                             - Electrical Cabling/Wiring: Includes cables, wires, and connectors.
                             - Electrical Connection/Distribution: Includes circuit breakers, fuse boxes, and switchboards.

                             Price Context Rules:
                             - Home Appliances > SAR 500 typically indicate Major Domestic Appliances
                             - Electrical Supplies > SAR 300 often involve professional-grade equipment
                             - Food/Beverage < SAR 20 are usually basic ingredients

                             Respond with JSON containing:
                             - segment_name (string)
                             - family_name (string)"""

                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                response = json.loads(chat_completion.choices[0].message.content)

                responses.append({
                    "item_name": item['item_name'],
                    "segment_code": self.segment_mapping.get(response.get('segment_name', ''), ''),
                    "segment_name": response.get('segment_name', ''),
                    "family_code": self.family_mapping.get(response.get('family_name', ''), ''),
                    "family_name": response.get('family_name', '')
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
        return responses

    def evaluate_model(self, model_name, sample):
        """Evaluation metrics calculation"""
        true_segments = sample['item_segment'].cat.codes.tolist()
        true_families = sample['item_family'].cat.codes.tolist()

        preds_segment = []
        preds_family = []
        inference_times = []

        for i in range(0, len(sample), self.batch_size):
            batch = sample.iloc[i:i+self.batch_size].to_dict('records')
            batch_preds, batch_time = self.process_batch(model_name, batch)

            preds_segment.extend([self.dataset['item_segment'].cat.categories.get_loc(p['segment_name'])
                                if p['segment_name'] in self.dataset['item_segment'].cat.categories else -1
                                for p in batch_preds])
            preds_family.extend([self.dataset['item_family'].cat.categories.get_loc(p['family_name'])
                               if p['family_name'] in self.dataset['item_family'].cat.categories else -1
                               for p in batch_preds])
            inference_times.append(batch_time)

        return {
            'accuracy_segment': accuracy_score(true_segments, preds_segment),
            'f1_segment': f1_score(true_segments, preds_segment, average='micro'),
            'accuracy_family': accuracy_score(true_families, preds_family),
            'f1_family': f1_score(true_families, preds_family, average='micro'),
            'avg_inference_time': np.mean(inference_times)
        }

    def run_pipeline(self, models, output_dir):
        """Main execution flow"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        eval_sample = self.dataset.sample(frac=0.2, random_state=42)

        for model in models:
            all_predictions = []
            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset.iloc[i:i+self.batch_size].to_dict('records')
                batch_preds, batch_time = self.process_batch(model, batch)

                all_predictions.extend([{
                    'Item_name': item['item_name'],
                    'true_segment': item['item_segment'],
                    'true_family': item['item_family'],
                    'pred_segment': pred['segment_name'],
                    'pred_family': pred['family_name'],
                    'inference_time': batch_time/len(batch)
                } for item, pred in zip(batch, batch_preds)])

            filename = f"{output_dir}/{model.replace('/','-')}_GROQ-LPU.csv"
            pd.DataFrame(all_predictions).to_csv(filename, index=False)

            results[model] = self.evaluate_model(model, eval_sample)

        with open(f"{output_dir}/performance.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.generate_detailed_report(all_predictions, output_dir)

        return results

    def generate_detailed_report(self, predictions, output_dir):
        df = pd.DataFrame(predictions)

        segment_errors = df[df['true_segment'] != df['pred_segment']]
        segment_error_counts = segment_errors.groupby(['true_segment', 'pred_segment']).size()

        family_errors = df[(df['true_segment'] == df['pred_segment']) & (df['true_family'] != df['pred_family'])]
        family_error_counts = family_errors.groupby(['true_family', 'pred_family']).size()

        low_confidence_predictions = df[df['pred_segment'] == 'Unknown']
        low_confidence_count = len(low_confidence_predictions)
        low_confidence_segment_distribution = low_confidence_predictions['true_segment'].value_counts()

        segment_errors_without_brand = segment_errors[segment_errors['true_segment'] == 'Unknown']
        segment_errors_without_brand_count = segment_errors_without_brand['true_segment'].value_counts()

        family_errors_with_specs = family_errors['true_family'].value_counts()

        with open(f"{output_dir}/performance.json", 'r') as f:
            performance_metrics = json.load(f)

        report = f"""
        === Segment-Level Errors ===
        {segment_error_counts}

        === Family-Level Errors (Within Correct Segments) ===
        {family_error_counts}

        === Low-Confidence Predictions ===
        Count: {low_confidence_count}
        Segment Distribution:
        {low_confidence_segment_distribution}

        === Error Features Analysis ===
        Segment errors without brand:
        {segment_errors_without_brand_count}

        Family errors with specs:
        {family_errors_with_specs}

        === Final Performance Metrics ===
        {json.dumps(performance_metrics, indent=2)}
        """

        with open(f"{output_dir}/detailed_report.txt", 'w') as f:
            f.write(report)

        print(report)

if __name__ == '__main__':
    # Install required packages
    #!pip install -qU pandas scikit-learn groq
    config = {
        'dataset_path': '/content/train_family_en.csv',
        'groq_api_key': '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',
        'models_to_test': ['gemma2-9b-it'],
        'output_dir': '/content/results/new_gemma_balance2'
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    processor = InferencePipeline(config['dataset_path'], config['groq_api_key'])
    performance = processor.run_pipeline(config['models_to_test'], config['output_dir'])
    print("\nPerformance Results:", json.dumps(performance, indent=2))