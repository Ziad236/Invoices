import os
import json
import time
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from groq import Groq


family_dict = {
    "Food/Beverage": {
        "Fruits/Vegetables/Nuts/Seeds Prepared/Processed": {
            "code": "50100000",
            "description": "Processed fruits, vegetables, nuts, and seeds, including dried, canned, and frozen. Example items: 'Canned Peaches', 'Frozen Mixed Vegetables', 'Dried Mango Slices'."
        },
        "Fish and Seafood": {
            "code": "50120000",
            "description": "Fresh and processed fish, shellfish, and seafood. Example items: 'Salmon Fillets', 'Canned Tuna', 'Frozen Shrimp'."
        },
        "Milk/Butter/Cream/Yogurts/Cheese/Eggs/Substitutes": {
            "code": "50130000",
            "description": "Dairy and substitutes, including milk, butter, yogurt, cheese, and eggs. Example items: 'Skim Milk', 'Vegan Cheese', 'Greek Yogurt'."
        },
        "Oils/Fats Edible": {
            "code": "50150000",
            "description": "Edible oils and fats for cooking, such as olive oil and butter. Example items: 'Olive Oil', 'Sunflower Oil', 'Cooking Margarine'."
        },
        "Confectionery/Sugar Sweetening Products": {
            "code": "50160000",
            "description": "Candies, chocolates, and sugar-based sweeteners. Example items: 'Chocolate Bars', 'Hard Candies', 'Brown Sugar'."
        },
        "Seasonings/Preservatives/Extracts": {
            "code": "50170000",
            "description": "Flavor enhancers, spices, herbs, and food preservatives. Example items: 'Salt', 'Mixed Spices', 'Vanilla Extract'."
        },
        "Bread/Bakery Products": {
            "code": "50180000",
            "description": "Bread, cakes, pastries, and other baked goods. Example items: 'White Bread Loaf', 'Chocolate Muffins', 'Croissants'."
        },
        "Prepared/Preserved Foods": {
            "code": "50190000",
            "description": "Ready-to-eat and preserved foods like canned goods and pickles. Example items: 'Canned Beans', 'Pickled Cucumbers', 'Instant Soup'."
        },
        "Beverages": {
            "code": "50200000",
            "description": "Non-alcoholic drinks, including juices, sodas, and water. Example items: 'Orange Juice', 'Carbonated Soda', 'Bottled Water'."
        },
        "Cereal/Grain/Pulse Products": {
            "code": "50220000",
            "description": "Cereal, grains, pasta, rice, and legumes. Example items: 'Corn Flakes', 'Spaghetti Pasta', 'Brown Rice'."
        },
        "Meat/Poultry/Other Animals": {
            "code": "50240000",
            "description": "Fresh and processed meat, poultry, and animal proteins. Example items: 'Chicken Breast', 'Ground Beef', 'Pork Sausages'."
        },
        "Vegetables - Unprepared/Unprocessed (Frozen)": {
            "code": "50290000",
            "description": "Frozen vegetables, raw and unprocessed. Example items: 'Frozen Peas', 'Frozen Broccoli Florets'."
        },
        "Fruits - Unprepared/Unprocessed (Frozen)": {
            "code": "50270000",
            "description": "Frozen fruits, raw and unprocessed. Example items: 'Frozen Strawberries', 'Frozen Blueberries'."
        },
        "Fruits - Unprepared/Unprocessed (Fresh)": {
            "code": "50250000",
            "description": "Fresh, raw, unprocessed fruits. Example items: 'Apples', 'Bananas', 'Grapes'."
        },
        "Fruits/Vegetables Fresh Cut": {
            "code": "50380000",
            "description": "Pre-cut fresh fruits and vegetables. Example items: 'Sliced Apples', 'Mixed Salad Greens', 'Diced Onions'."
        },
        "Meat/Fish/Seafood Substitutes": {
            "code": "50390000",
            "description": "Alternatives to meat, fish, or seafood, like tofu. Example items: 'Tofu', 'Tempeh', 'Vegan Burgers'."
        },
        "Vegetables - Unprepared/Unprocessed (Shelf Stable)": {
            "code": "50320000",
            "description": "Shelf-stable raw vegetables. Example items: 'Dried Mushrooms', 'Vacuum-Packed Potatoes'."
        },
        "Fruits - Unprepared/Unprocessed (Shelf Stable)": {
            "code": "50310000",
            "description": "Shelf-stable raw fruits, including dried varieties. Example items: 'Dried Apricots', 'Dehydrated Apple Slices'."
        },
        "Nuts/Seeds - Unprepared/Unprocessed (Perishable)": {
            "code": "50330000",
            "description": "Perishable raw nuts and seeds. Example items: 'Raw Peanuts', 'Raw Almonds'."
        },
        "Nuts/Seeds - Unprepared/Unprocessed (In Shell)": {
            "code": "50340000",
            "description": "Raw nuts and seeds in shells. Example items: 'Peanuts in Shell', 'Sunflower Seeds in Shell'."
        },
        "Fruits/Vegetables Fresh & Fresh Cut": {
            "code": "50370000",
            "description": "Whole and pre-cut fresh produce. Example items: 'Whole Carrots', 'Chopped Bell Peppers'."
        },
        "Vegetables (Non Leaf) - Unprepared/Unprocessed (Fresh)": {
            "code": "50260000",
            "description": "Fresh, raw non-leafy vegetables. Example items: 'Potatoes', 'Carrots', 'Bell Peppers'."
        },
        "Leaf Vegetables - Unprepared/Unprocessed (Fresh)": {
            "code": "50350000",
            "description": "Fresh, raw leafy vegetables. Example items: 'Lettuce', 'Spinach'."
        }
    },
    "Healthcare": {
        "Family Planning": {
            "code": "51110000",
            "description": "Contraceptives and reproductive health products. Example items: 'Birth Control Pills', 'Condoms'."
        },
        "Health Enhancement": {
            "code": "51120000",
            "description": "Wellness products, including vitamins and supplements. Example items: 'Multivitamins', 'Herbal Supplements'."
        },
        "Medical Devices": {
            "code": "51150000",
            "description": "Diagnostic and therapeutic devices. Example items: 'Blood Pressure Monitor', 'Nebulizer'."
        },
        "Pharmaceutical Drugs": {
            "code": "51160000",
            "description": "Prescription and over-the-counter medications. Example items: 'Ibuprofen Tablets', 'Antibiotics'."
        },
        "Health Treatments/Aids": {
            "code": "51100000",
            "description": "Therapeutic aids, including bandages and braces. Example items: 'Elastic Bandages', 'Knee Braces'."
        },
        "Veterinary Healthcare": {
            "code": "51170000",
            "description": "Animal healthcare products, including pet medications. Example items: 'Flea & Tick Prevention', 'Deworming Tablets'."
        },
        "Home Diagnostics": {
            "code": "51130000",
            "description": "At-home diagnostic and testing kits. Example items: 'Blood Glucose Test Strips', 'COVID-19 Rapid Test Kits'."
        }
    },
    "Clothing": {
        "Clothing": {
            "code": "67010000",
            "description": "Everyday wear, including tops, bottoms, and outerwear. Example items: 'Jeans', 'T-Shirts', 'Jackets'."
        },
        "Activewear": {
            "code": "67030000",
            "description": "Sportswear and fitness clothing. Example items: 'Running Shorts', 'Yoga Pants'."
        },
        "Underwear": {
            "code": "67040000",
            "description": "Undergarments, including bras and socks. Example items: 'Briefs', 'Sports Bras'."
        },
        "Swimwear": {
            "code": "67060000",
            "description": "Swimsuits and beachwear. Example items: 'Bikini', 'Swim Trunks'."
        },
        "Protective Wear": {
            "code": "67050000",
            "description": "Safety and protective clothing. Example items: 'Reflective Vests', 'Hazmat Suits'."
        },
        "Sleepwear": {
            "code": "67020000",
            "description": "Pajamas and loungewear. Example items: 'Nightgown', 'Flannel Pajamas'."
        }
    },
    "Home Appliances": {
        "Major Domestic Appliances": {
            "code": "72010000",
            "description": "Large household appliances, such as refrigerators and ovens. Example items: 'Refrigerator', 'Gas Oven'."
        },
        "Small Domestic Appliances": {
            "code": "72020000",
            "description": "Smaller appliances, such as microwaves and vacuum cleaners. Example items: 'Microwave Oven', 'Handheld Vacuum'."
        }
    },
    "Electrical Supplies": {
        "Electrical Connection/Distribution": {
            "code": "78020000",
            "description": "Power distribution and electrical connection components. Example items: 'Power Strips', 'Circuit Breakers'."
        },
        "Electrical Lighting": {
            "code": "78030000",
            "description": "Indoor and outdoor lighting solutions. Example items: 'LED Bulbs', 'Desk Lamps'."
        },
        "Electrical Cabling/Wiring": {
            "code": "78040000",
            "description": "Cables and wiring for electrical installations. Example items: 'Ethernet Cables', 'Electrical Wire Reels'."
        },
        "General Electrical Hardware": {
            "code": "78060000",
            "description": "General electrical components, including switches and connectors. Example items: 'Wall Switches', 'Electrical Outlets'."
        },
        "Electronic Communication Components": {
            "code": "78050000",
            "description": "Electronic communication system components, such as routers. Example items: 'Wi-Fi Router', 'Network Switch'."
        }
    },
    "Computing": {
        "Computers/Video Games": {
            "code": "65010000",
            "description": "Computers, gaming consoles, and accessories. Example items: 'Desktop PC', 'PlayStation Console', 'Gaming Mouse'."
        }
    },
    "Tools/Equipment": {
        "Tools/Equipment": {
            "code": "74010000",
            "description": "Hand tools, power tools, and industrial equipment. Example items: 'Hammer', 'Electric Drill', 'Wrench Set'."
        }
    },
    "Camping": {
        "Camping": {
            "code": "75010000",
            "description": "Camping and outdoor recreational equipment. Example items: 'Tent', 'Sleeping Bag', 'Camping Stove'."
        }
    },
    "Arts/Crafts/Needlework": {
        "Arts/Crafts/Needlework Supplies": {
            "code": "70010000",
            "description": "Materials and supplies for arts, crafts, and needlework. Example items: 'Yarn', 'Paint Brushes', 'Canvas'."
        }
    },
    "Music": {
        "Musical Instruments/Accessories": {
            "code": "61010000",
            "description": "Musical instruments and performance accessories. Example items: 'Acoustic Guitar', 'Drum Sticks', 'Violin'."
        }
    }
}

class InferencePipeline:
    def __init__(self, dataset_path, groq_api_key):
        self.batch_size = 3
        self.dataset = self.load_dataset(dataset_path)
        self.groq_client = Groq(api_key=groq_api_key)
        self.all_inferences = []

        self.family_details = family_dict

        self.segment_mapping = {}
        self.family_mapping = {}
        self.segment_to_families = {}
        self.family_descriptions = {}

        for segment, families in self.family_details.items():
            if families:
                base_code = next(iter(families.values()))['code'][:6]
                self.segment_mapping[segment] = f"{base_code}00"

            family_list = []
            for family, details in families.items():
                self.family_mapping[family] = details['code']
                self.family_descriptions[family] = details['description']
                family_list.append(family)

            self.segment_to_families[segment] = family_list

        self.candidate_labels = {
            'segments': list(self.segment_mapping.keys()),
            'families': list(self.family_mapping.keys()),
        }

        self.system_message = {
            "role": "system",
            "content": (
                "You are a classifier. You have a dictionary of segments and families, with their codes, descriptions, and example items. "
                "You will first classify items into one of these segments, then classify them into the correct family within that segment. "
                "Use the following reference to guide your classification:\n\n"
                + json.dumps(self.family_details, indent=2) +
                "\n\n"
                "When classifying segments or families, respond in valid JSON only. "
                "Do not add explanations. Only output JSON with the structure:\n"
                '{\n'
                '  "classifications": [\n'
                '    {"item_name": "...", "segment_name": "..." }  // for segment classification\n'
                '  ]\n'
                '}\n'
                "or:\n"
                '{\n'
                '  "classifications": [\n'
                '    {"item_name": "...", "family_name": "..." }   // for family classification\n'
                '  ]\n'
                '}'
            )
        }

        self.total_requests = 0
        self.progress_bar = None

    def load_dataset(self, path):
        """Load, shuffle, and limit dataset to 2000 samples (or 1900)."""
        required_columns = ['item_name', 'item_segment', 'item_family']
        df = pd.read_csv(path, encoding='utf-8 sig')
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df['item_segment'] = df['item_segment'].fillna('Unknown').astype('category')
        df['item_family'] = df['item_family'].fillna('Unknown').astype('category')
        df = df.sample(frac=1, random_state=42).head(1900).reset_index(drop=True)

        return df

    def process_batch(self, batch, model_name):
        """
        Process a batch with hierarchical classification:
          1) Classify segments
          2) Group items by predicted segment
          3) Classify families for each segment
        """
        start_time = time.time()
        items = [item['item_name'] for item in batch]
        segment_mapping = self._classify_segments(items, model_name)
        segment_groups = self._group_by_segment(batch, segment_mapping)
        family_mapping = self._classify_families(segment_groups, model_name)

        if self.progress_bar:
            self.progress_bar.update(len(batch))

        return self._compile_results(batch, segment_mapping, family_mapping, start_time)

    def _classify_segments(self, items, model_name):
        """
        Classify items into segments (one call for all items in this batch).
        We DO pass the self.system_message here.
        """
        prompt = self._create_segment_prompt(items)
        max_retries = 3
        delay = 2

        for attempt in range(max_retries):
            try:
                self.total_requests += 1
                response = self.groq_client.chat.completions.create(
                    messages=[
                        self.system_message,
                        {"role": "user", "content": prompt}
                    ],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                time.sleep(random.uniform(2, 3))
                return self._parse_classification_response(
                    response,
                    label_key="segment_name",
                    valid_options=self.candidate_labels['segments']
                )
            except Exception as e:
                print(f"Attempt {attempt + 1} for segment classification failed: {e}")
                time.sleep(delay)
                delay *= 2  

        print("Segment classification failed after retries.")
        return {}

    def _classify_families(self, segment_groups, model_name):
        """
        Classify items into families, one segment at a time.
        Also uses the single system message to avoid re-sending dict each time.
        """
        family_mapping = {}

        for segment, group in segment_groups.items():
            if not segment:
                continue

            items = [item['item_name'] for item in group]
            prompt = self._create_family_prompt(items, segment)
            max_retries = 3
            delay = 2

            for attempt in range(max_retries):
                try:
                    self.total_requests += 1
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            self.system_message, 
                            {"role": "user", "content": prompt}
                        ],
                        model=model_name,
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    time.sleep(random.uniform(2, 3))

                    family_mapping.update(
                        self._parse_classification_response(
                            response,
                            label_key="family_name",
                            valid_options=self.segment_to_families.get(segment, [])
                        )
                    )
                    break  
                except Exception as e:
                    print(f"Family classification attempt {attempt + 1} failed for segment '{segment}': {e}")
                    time.sleep(delay)
                    delay *= 2

        return family_mapping

    def _parse_classification_response(self, response, label_key, valid_options):
        """
        Parse the JSON response from the model. We only accept recognized labels.
        If the model's output label is not in 'valid_options', we skip or mark it as ''.
        """
        try:
            data = json.loads(response.choices[0].message.content)
            results = {}
            for item in data.get('classifications', []):
                predicted_label = item.get(label_key, '')
                if predicted_label in valid_options:
                    results[item['item_name']] = predicted_label
                else:
                    results[item['item_name']] = ''  
            return results
        except Exception as e:
            print(f"Parsing error: {e}")
            return {}

    def _group_by_segment(self, batch, segment_mapping):
        """
        Group items by predicted segment so we can classify families segment-by-segment.
        """
        groups = defaultdict(list)
        for item in batch:
            predicted_segment = segment_mapping.get(item['item_name'], '')
            groups[predicted_segment].append(item)
        return groups

    def _compile_results(self, batch, segment_mapping, family_mapping, start_time):
        """
        Build a final structure with segment_code, family_code, etc.
        """
        responses = []
        for item in batch:
            seg_name = segment_mapping.get(item['item_name'], 'Unknown')
            fam_name = family_mapping.get(item['item_name'], 'Unknown')
            responses.append({
                "item_name": item['item_name'],
                "segment_code": self.segment_mapping.get(seg_name, ''),
                "segment_name": seg_name,
                "family_code": self.family_mapping.get(fam_name, ''),
                "family_name": fam_name
            })
        avg_time = (time.time() - start_time) / len(batch) if batch else 0
        return responses, avg_time

    def _create_segment_prompt(self, items):
        """
        Create a classification prompt for segments.
        We do NOT re-send the entire dictionary here, only instructions that
        the model has in system_message. We just specify the items and remind
        the model to classify them into known segments.
        """
        items_list = "\n".join([f"- {item}" for item in items])
        return (
            f"Classify these products into one of the known segments:\n"
            f"{items_list}\n\n"
            "Output JSON with the structure:\n"
            '{"classifications": [{"item_name": "...", "segment_name": "..."}]}\n'
        )

    def _create_family_prompt(self, items, segment):
        """
        Create a classification prompt for families within a given segment.
        The system message already contains the dictionary, so we just specify the segment and items.
        """
        items_list = "\n".join([f"- {item}" for item in items])
        return (
            f"The predicted segment is '{segment}'. "
            "Now classify these products into the correct family for that segment:\n"
            f"{items_list}\n\n"
            "Output JSON with the structure:\n"
            '{"classifications": [{"item_name": "...", "family_name": "..."}]}\n'
        )

    def evaluate_model(self, model_name, sample):
        """
        Evaluate the model predictions against true segments/families
        using simple accuracy and F1 (micro).
        """
        true_segments = sample['item_segment'].cat.codes.tolist()
        true_families = sample['item_family'].cat.codes.tolist()

        preds_segment = []
        preds_family = []
        inference_times = []

        for i in range(0, len(sample), self.batch_size):
            batch = sample.iloc[i:i + self.batch_size].to_dict('records')
            batch_preds, batch_time = self.process_batch(batch, model_name)

            for bp in batch_preds:
                seg_code = sample['item_segment'].cat.categories
                fam_code = sample['item_family'].cat.categories

                if bp['segment_name'] in seg_code:
                    preds_segment.append(seg_code.get_loc(bp['segment_name']))
                else:
                    preds_segment.append(-1)

                if bp['family_name'] in fam_code:
                    preds_family.append(fam_code.get_loc(bp['family_name']))
                else:
                    preds_family.append(-1)

            inference_times.append(batch_time)

        return {
            'accuracy_segment': accuracy_score(true_segments, preds_segment),
            'f1_segment': f1_score(true_segments, preds_segment, average='micro'),
            'accuracy_family': accuracy_score(true_families, preds_family),
            'f1_family': f1_score(true_families, preds_family, average='micro'),
            'avg_inference_time': np.mean(inference_times)
        }

    def run_pipeline(self, models, output_dir):
        """
        Run the pipeline for each model:
          1) Classify entire dataset in batches, store predictions
          2) Save predictions to a CSV
          3) Evaluate
          4) Print results
        """
        os.makedirs(output_dir, exist_ok=True)
        self.progress_bar = tqdm(total=len(self.dataset), desc="Classifying Items", unit="items")

        for model in models:
            self.all_inferences.clear()
            self.progress_bar.reset()
            self.progress_bar.total = len(self.dataset)
            self.progress_bar.refresh()

            for i in range(0, len(self.dataset), self.batch_size):
                batch = self.dataset.iloc[i:i + self.batch_size].to_dict('records')
                batch_preds, batch_time = self.process_batch(batch, model)
                for bp in batch_preds:
                    bp["batch_inference_time"] = batch_time
                    self.all_inferences.append(bp)

            results_df = pd.DataFrame(self.all_inferences)
            csv_path = os.path.join(output_dir, f"{model}_predictions.csv")
            results_df.to_csv(csv_path, index=False)

            results = self.evaluate_model(model, self.dataset)
            print(f"\n=== Evaluation results for model '{model}' ===")
            print(json.dumps(results, indent=2), "\n")

        self.progress_bar.close()
        print(f"Pipeline complete! Total API requests sent: {self.total_requests}")

if __name__ == '__main__':
    config = {
        'dataset_path': '/content/arabic_test_data.csv',    
        'groq_api_key': 'YOUR_GROQ_API_KEY_HERE',        
        'models_to_test': ['gemma2-9b-it'],            
        'output_dir': '/content/new_Approach_results'
    }

    pipeline = InferencePipeline(config['dataset_path'], config['groq_api_key'])
    pipeline.run_pipeline(config['models_to_test'], config['output_dir'])

 
    # from google.colab import files
    # try:
    #     csv_path = os.path.join(config['output_dir'], f"{config['models_to_test'][0]}_predictions.csv")
    #     files.download(csv_path)
    # except Exception as e:
    #     print(f"Download failed: {e}")
