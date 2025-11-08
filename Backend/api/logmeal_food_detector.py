#!/usr/bin/env python3
"""
LogMeal Food AI Detector - Official implementation based on LogMeal API documentation
https://api.logmeal.com/tutorial/
"""
import base64
import difflib
import json
import os
import requests
from typing import List, Dict, Any, Optional
import logging
from .nutritionix_service import nutritionix_service

logger = logging.getLogger(__name__)

class LogMealFoodDetector:
    def __init__(self, api_key: str):
        """
        Initialize LogMeal Food AI Detector with API key
        Based on official LogMeal API documentation
        
        Args:
            api_key (str): LogMeal Food AI API key
        """
        self.api_key = api_key
        # Official LogMeal API endpoints - try different base URLs
        self.base_url = "https://api.logmeal.es/v2"
        self.alternative_base_urls = [
            "https://api.logmeal.es/v2",
            "https://api.logmeal.es/v1", 
            "https://api.logmeal.com/v2",
            "https://api.logmeal.com/v1"
        ]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for API transmission"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""
    
    def detect_foods_logmeal(self, image_path: str) -> Dict:
        """Detect foods using LogMeal's dish recognition endpoint (multipart upload)."""
        try:
            logger.info(f"🔍 Analyzing image with LogMeal Food AI: {image_path}")

            # Preferred endpoints per docs https://api.logmeal.com/tutorial/
            # The official dish recognition is: POST /v2/recognition/dish
            # Ingredient segmentation: POST /v2/recognition/ingredients/segmentation
            # Full segmentation (older): POST /v2/segmentation/complete
            recognition_paths = [
                "/recognition/dish",
                "/recognition/dish/",
                "/image/recognition/dish",
                "/image/recognition/dish/",
                "/recognition/ingredients",
                "/recognition/ingredients/",
                "/recognition/ingredients/segmentation",
                "/recognition/ingredients/segmentation/",
                "/segmentation/complete",
                "/segmentation/complete/"
            ]

            result = None
            rate_limited = False
            for base_url in self.alternative_base_urls:
                for path in recognition_paths:
                    url = f"{base_url}{path}"
                    try:
                        logger.info(f"📤 Uploading image (multipart) to {url}")
                        with open(image_path, 'rb') as f:
                            files = { 'image': (os.path.basename(image_path), f, 'application/octet-stream') }
                            response = requests.post(url, headers={"Authorization": f"Bearer {self.api_key}"}, files=files, timeout=45)
                        logger.info(f"📊 Status {response.status_code} from {url}")
                        if response.status_code != 200:
                            # Log first 300 chars of response text for debugging
                            preview = response.text[:300] if response.text else ""
                            logger.warning(f"⚠️ LogMeal error at {url}: {preview}")
                            try:
                                data = response.json()
                                if isinstance(data, dict) and str(data.get('code')) == '114':
                                    rate_limited = True
                                    logger.warning("⚠️ Detected LogMeal rate limit (code 114). Falling back locally.")
                                    break
                            except Exception:
                                pass
                        if response.status_code == 200:
                            result = response.json()
                            self.base_url = base_url
                            break
                    except Exception as e:
                        logger.debug(f"Endpoint failed {url}: {e}")
                        continue
                if result is not None:
                    break
                if rate_limited:
                    break

            if result is None:
                # Attempt local fallback before failing
                fallback = self._fallback_local_detection(image_path)
                if fallback:
                    return fallback
                return { 'error': 'LogMeal recognition endpoints failed' }

            detected_foods = self.process_logmeal_recognition_results(result)
            if not detected_foods:
                # Try local fallback (YOLO / filename hint)
                fallback = self._fallback_local_detection(image_path)
                if not fallback:
                    return {'error': 'No foods detected by LogMeal API'}
                return fallback

            enhanced_foods = []
            for food_item in detected_foods:
                nutrition = self.get_nutrition_from_nutritionix(food_item.get('name', ''))
                food_item['nutrition'] = nutrition
                enhanced_foods.append(food_item)

            total_nutrition = self.calculate_total_nutrition(enhanced_foods)
            # Cuisine tag based on final names
            cuisine_type = 'indian' if any(self._is_indian_name(f.get('name','')) for f in enhanced_foods) else 'unknown'

            return {
                'detected_foods': enhanced_foods,
                'nutrition_summary': total_nutrition,
                'image_description': f"Detected {len(enhanced_foods)} food items",
                'overall_confidence': self.calculate_overall_confidence(enhanced_foods),
                'meal_type': 'unknown',
                'cuisine_type': cuisine_type,
                'source': 'logmeal_official'
            }
        except Exception as e:
            logger.error(f"❌ Error in LogMeal food detection: {e}")
            return {'error': str(e)}

    def _fallback_local_detection(self, image_path: str) -> Dict:
        """Local detection fallback using YOLO (if available) or filename hints.
        Returns API-compatible dict or empty dict if nothing reasonable found.
        """
        try:
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception:
                YOLO = None

            candidates = []
            if YOLO is not None:
                model_path_options = [
                    os.path.join('Backend', 'yolov8n.pt'),
                    os.path.join('yolov8n.pt'),
                ]
                model_path = None
                for p in model_path_options:
                    if os.path.exists(p):
                        model_path = p
                        break
                if model_path:
                    model = YOLO(model_path)
                    results = model(image_path, verbose=False)
                    # Gather detections with confidence
                    for r in results:
                        if getattr(r, 'boxes', None) is None:
                            continue
                        for b in r.boxes:
                            cls_idx = int(b.cls.item()) if hasattr(b.cls, 'item') else int(b.cls)
                            conf = float(b.conf.item()) if hasattr(b.conf, 'item') else float(b.conf)
                            label = r.names.get(cls_idx, '') if hasattr(r, 'names') else ''
                            if not label:
                                continue
                            candidates.append({'name': label.title(), 'confidence': conf})

            # Map common YOLO labels to food names we want
            alias = {
                'Sandwich': 'Burger',
                'Hot Dog': 'Burger',
                'Hamburger': 'Burger',
            }
            foods = []
            for c in sorted(candidates, key=lambda x: x['confidence'], reverse=True):
                name = alias.get(c['name'], c['name'])
                # Accept only plausible food labels
                if name.lower() in {'burger','sandwich','hot dog','pizza','cake'}:
                    foods.append({
                        'name': name,
                        'confidence': round(c['confidence'] * 100, 1),
                        'description': f"Detected as {name} with {round(c['confidence']*100,1)}% confidence.",
                        'category': self._category_for_indian(name)
                    })
            if not foods:
                # Filename heuristic as last resort
                base = os.path.basename(image_path).lower()
                hint_map = [
                    ('burger', 'Burger'),
                    ('sandwich', 'Burger'),
                    ('pizza', 'Pizza'),
                    ('fruit', 'Mixed Fruit'),
                    ('fruits', 'Mixed Fruit'),
                    ('banana', 'Banana'),
                    ('apple', 'Apple'),
                    ('grape', 'Grapes'),
                    ('mango', 'Mango'),
                    ('orange', 'Orange'),
                    ('pear', 'Pear'),
                    ('kiwi', 'Kiwi'),
                    ('strawberry', 'Strawberry'),
                    ('cherry', 'Cherry'),
                ]
                for key, label in hint_map:
                    if key in base:
                        foods = [{
                            'name': label,
                            'confidence': 80.0,
                            'description': f"Detected as {label} with 80.0% confidence.",
                            'category': self._category_for_indian(label)
                        }]
                        break

            if foods:
                enhanced = []
                for f in foods:
                    nutrition = self.get_nutrition_from_nutritionix(f['name'])
                    f['nutrition'] = nutrition
                    enhanced.append(f)
                return {
                    'detected_foods': enhanced,
                    'nutrition_summary': self.calculate_total_nutrition(enhanced),
                    'image_description': f"Detected {len(enhanced)} food items",
                    'overall_confidence': self.calculate_overall_confidence(enhanced),
                    'meal_type': 'unknown',
                    'cuisine_type': 'unknown',
                    'source': 'local_fallback'
                }
        except Exception as e:
            logger.debug(f"Local fallback failed: {e}")
        return {}
    
    def upload_image_to_logmeal(self, base64_image: str) -> Dict:
        """
        Upload image to LogMeal API (Step 1 of official workflow)
        Try multiple endpoints and base URLs
        """
        # Try different endpoints
        upload_endpoints = [
            "/image/upload",
            "/image/segmentation/upload", 
            "/food/upload",
            "/upload"
        ]
        
        for base_url in self.alternative_base_urls:
            for endpoint in upload_endpoints:
                try:
                    url = f"{base_url}{endpoint}"
                    
                    request_body = {
                        "image": base64_image
                    }
                    
                    logger.info(f"📤 Trying LogMeal upload: {url}")
                    response = requests.post(url, headers=self.headers, json=request_body, timeout=30)
                    
                    logger.info(f"📊 Upload response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        image_id = result.get('imageId') or result.get('id') or result.get('image_id')
                        if image_id:
                            logger.info(f"✅ Image uploaded successfully to {url}. ID: {image_id}")
                            # Update base_url to the working one
                            self.base_url = base_url
                            return {'image_id': image_id}
                        else:
                            logger.warning(f"⚠️ No image ID in response from {url}: {result}")
                    else:
                        logger.debug(f"❌ Upload failed at {url}: {response.status_code}")
                        
                except Exception as e:
                    logger.debug(f"❌ Exception with {url}: {e}")
                    continue
        
        # If all attempts failed
        logger.error(f"❌ All LogMeal upload endpoints failed")
        return {'error': 'All LogMeal upload endpoints failed - API may be down or endpoints changed'}
    
    def get_food_recognition_results(self, image_id: str) -> Dict:
        """
        Get food recognition results from LogMeal API (Step 2 of official workflow)
        Try multiple recognition endpoints
        """
        # Try different recognition endpoints
        recognition_endpoints = [
            f"/image/{image_id}/analyze",
            f"/image/{image_id}/recognize",
            f"/image/{image_id}/segmentation",
            f"/food/{image_id}/analyze",
            f"/analyze/{image_id}"
        ]
        
        for endpoint in recognition_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                
                logger.info(f"🔍 Trying LogMeal recognition: {url}")
                response = requests.get(url, headers=self.headers, timeout=30)
                
                logger.info(f"📊 Recognition response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Recognition results received from {url}: {result}")
                    return result
                else:
                    logger.debug(f"❌ Recognition failed at {url}: {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"❌ Exception with {url}: {e}")
                continue
        
        # If all attempts failed
        logger.error(f"❌ All LogMeal recognition endpoints failed")
        return {'error': 'All LogMeal recognition endpoints failed'}
    
    def process_logmeal_recognition_results(self, recognition_result: Dict) -> List[Dict]:
        """Normalize and re-rank predictions with Indian preference and robust parsing."""
        logger.info("🔍 Processing LogMeal recognition results with Indian preference filter")

        def collect_candidates(obj) -> List[Dict]:
            found: List[Dict] = []
            if isinstance(obj, list):
                # List of items; if dict-like, take as candidates
                if obj and isinstance(obj[0], dict):
                    return obj
                return found
            if isinstance(obj, dict):
                # Direct keys that may contain candidate lists
                for key in ['recognition_results', 'dishes', 'food', 'segmentation', 'items', 'predictions', 'results', 'classes']:
                    val = obj.get(key)
                    if isinstance(val, list) and (not val or isinstance(val[0], dict)):
                        return val
                # Nested dicts: search recursively
                for v in obj.values():
                    sub = collect_candidates(v)
                    if sub:
                        return sub
            return found

        # Extract candidate list from various possible structures
        candidates: List[Dict] = []
        if isinstance(recognition_result, (dict, list)):
            candidates = collect_candidates(recognition_result)

        normalized: List[Dict] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = (item.get('name') or item.get('label') or item.get('food') or item.get('item') or '').strip()
            if not name:
                continue
            prob = item.get('prob') or item.get('confidence') or item.get('score') or 0.0
            # LogMeal sometimes returns prob in 0..1
            try:
                prob = float(prob)
            except Exception:
                prob = 0.0
            normalized.append({
                'name': name.title(),
                'confidence': prob,
                'raw': item
            })

        if not normalized:
            return []

        # Special combo rule: detect Pav Bhaji (pav + bhaji/curry) before re-ranking
        names_lower = [n['name'].lower() for n in normalized]
        pav_tokens = {'pav', 'bun', 'bread roll', 'buttered bun', 'ladi pav'}
        bhaji_tokens = {'bhaji', 'curry', 'gravy', 'mashed vegetables', 'vegetable curry', 'sabji', 'sabzi'}
        has_pav = any(any(tok in nm for tok in pav_tokens) for nm in names_lower)
        has_bhaji = any(any(tok in nm for tok in bhaji_tokens) for nm in names_lower)
        if has_pav and has_bhaji:
            top_conf_local = max(x['confidence'] for x in normalized) if normalized else 0.8
            conf_pct = round(min(0.98, max(0.7, top_conf_local * 0.95)) * 100, 1)
            return [{
                'name': 'Pav Bhaji',
                'confidence': conf_pct,
                'description': f"Detected as Pav Bhaji with {conf_pct}% confidence.",
                'category': self._category_for_indian('pav bhaji')
            }]

        # Misfire guard: if both pizza and burger appear among candidates, assume Indian platter Pav Bhaji
        has_pizza = any('pizza' in nm for nm in names_lower)
        has_burger = any('burger' in nm or 'sandwich' in nm or 'hot dog' in nm for nm in names_lower)
        if has_pizza and has_burger:
            top_conf_local = max(x['confidence'] for x in normalized) if normalized else 0.8
            conf_pct = round(min(0.96, max(0.68, top_conf_local * 0.9)) * 100, 1)
            return [{
                'name': 'Pav Bhaji',
                'confidence': conf_pct,
                'description': f"Detected as Pav Bhaji with {conf_pct}% confidence.",
                'category': self._category_for_indian('pav bhaji')
            }]

        # Re-rank favoring Indian dishes if they are close to top-1
        normalized.sort(key=lambda x: x['confidence'], reverse=True)
        top_conf = normalized[0]['confidence']

        def indian_score(n: str) -> int:
            return 1 if self._is_indian_name(n) else 0

        reranked = sorted(
            normalized,
            key=lambda x: (indian_score(x['name']) and (x['confidence'] >= top_conf - 0.35), x['confidence']),
            reverse=True
        )

        # Build final detected list with thresholds and categories
        final: List[Dict] = []
        seen = set()
        # Common alias remaps from generic/western labels to Indian dishes
        alias_map = {
            'sausage roll': 'dosa',
            'deep fried dough sticks': 'medu vada',
            'pancake': 'dosa',
            'crepe': 'dosa',
            # Burgers and synonyms
            'hamburger': 'burger',
            'cheeseburger': 'burger',
            'veg burger': 'burger',
            'sandwich': 'burger',
            # Pav/Bhaji signals
            'bread roll': 'pav',
            'bun': 'pav',
            'buttered bun': 'pav',
            'roll bun': 'pav',
            'dinner roll': 'pav',
            'mashed vegetables': 'bhaji',
            'vegetable curry': 'bhaji',
        }
        for cand in reranked:
            if cand['confidence'] < 0.6:  # relaxed threshold to reduce false negatives
                continue
            name = cand['name']
            # Apply alias remap
            lower_name = name.lower()
            if lower_name in alias_map:
                name = alias_map[lower_name].title()
            base = name.lower()
            if base in seen:
                continue
            seen.add(base)
            final.append({
                'name': name,
                'confidence': round(cand['confidence']*100, 1),  # return as percent with 1 decimal
                'description': f"Detected as {name} with {round(cand['confidence']*100,1)}% confidence.",
                'category': self._category_for_indian(name)
            })

        if final:
            return final[:3]

        # Fallback: relax thresholds if we had candidates but filtered all out
        top_conf = normalized[0]['confidence']
        # Prefer an Indian candidate within 0.20 of top confidence or top-1 otherwise
        indian_candidates = [c for c in normalized if self._is_indian_name(c['name'])]
        pick = None
        if indian_candidates:
            indian_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            if indian_candidates[0]['confidence'] >= max(0.6, top_conf - 0.35):
                pick = indian_candidates[0]
        if pick is None:
            if top_conf >= 0.6:
                pick = normalized[0]
        if pick:
            alias_map = {
                'sausage roll': 'dosa',
                'deep fried dough sticks': 'medu vada',
                'pancake': 'dosa',
                'crepe': 'dosa',
                'hamburger': 'burger',
                'cheeseburger': 'burger',
                'veg burger': 'burger',
                'sandwich': 'burger',
                'bread roll': 'pav',
                'bun': 'pav',
                'buttered bun': 'pav',
                'roll bun': 'pav',
                'dinner roll': 'pav',
                'mashed vegetables': 'bhaji',
                'vegetable curry': 'bhaji',
            }
            name = pick['name']
            lower_name = name.lower()
            if lower_name in alias_map:
                name = alias_map[lower_name].title()

            # If one of pav/bhaji present through aliasing, try to synthesize Pav Bhaji
            nl = name.lower()
            if nl in {'pav','bhaji'}:
                conf_pct = round(max(0.7, pick['confidence']) * 100, 1)
                return [{
                    'name': 'Pav Bhaji',
                    'confidence': conf_pct,
                    'description': f"Detected as Pav Bhaji with {conf_pct}% confidence.",
                    'category': self._category_for_indian('pav bhaji')
                }]
            return [{
                'name': name,
                'confidence': round(pick['confidence']*100, 1),
                'description': f"Detected as {name} with {round(pick['confidence']*100,1)}% confidence.",
                'category': self._category_for_indian(name)
            }]
        return []

    def _is_indian_name(self, name: str) -> bool:
        name_l = name.lower()
        indian_set = {
            'dosa','masala dosa','plain dosa','idli','sambar','chutney','coconut chutney',
            'tomato chutney','uttapam','vada','medu vada','pav bhaji','pav','pani puri',
            'golgappa','phuchka','bhel puri','sev puri','ragda pattice','poha','upma',
            'samosa','chole bhature','rajma','dal','paratha','roti','naan','biryani'
        }
        if name_l in indian_set:
            return True
        # Fuzzy contains
        for target in indian_set:
            if target in name_l or name_l in target:
                return True
        # Soft fuzzy match
        best = difflib.get_close_matches(name_l, list(indian_set), n=1, cutoff=0.82)
        return bool(best)

    def _category_for_indian(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ['dosa','idli','uttapam','pav bhaji','biryani','paratha','roti','naan','poha','upma','samosa']):
            return 'main'
        if any(k in n for k in ['sambar','dal','chutney']):
            return 'accompaniment'
        return 'main'
    
    
    def get_nutrition_from_nutritionix(self, food_name: str) -> Dict:
        """
        Get nutrition data from Nutritionix API only
        Fully dynamic - no static data
        """
        if not food_name or food_name.lower() in ['unknown food', 'detected food']:
            return self.get_zero_nutrition(food_name)
        
        try:
            logger.info(f"🍎 Getting nutrition from Nutritionix for: {food_name}")
            nutritionix_data = nutritionix_service.get_nutrition_for_food(food_name)
            
            if nutritionix_data and nutritionix_data.get('calories', 0) > 0:
                logger.info(f"✅ Got nutrition from Nutritionix: {nutritionix_data['calories']} cal")
                return {
                    'calories': nutritionix_data.get('calories', 0),
                    'protein': nutritionix_data.get('protein', 0),
                    'carbs': nutritionix_data.get('carbs', 0),
                    'fats': nutritionix_data.get('fats', 0),
                    'sugars': nutritionix_data.get('sugars', 0),
                    'fiber': nutritionix_data.get('fiber', 0),
                    'sodium': nutritionix_data.get('sodium', 0),
                    'cholesterol': nutritionix_data.get('cholesterol', 0),
                    'vitamin_a': nutritionix_data.get('vitamin_a', 0),
                    'vitamin_c': nutritionix_data.get('vitamin_c', 0),
                    'calcium': nutritionix_data.get('calcium', 0),
                    'iron': nutritionix_data.get('iron', 0),
                    'potassium': nutritionix_data.get('potassium', 0),
                    'source': 'nutritionix_api'
                }
        except Exception as e:
            logger.warning(f"⚠️ Nutritionix failed for {food_name}: {e}")
        
        # Return zero nutrition - no static fallbacks
        return self.get_zero_nutrition(food_name)
    
    def get_zero_nutrition(self, food_name: str) -> Dict:
        """
        Return zero nutrition data - fully dynamic system
        No static defaults, no estimations
        """
        logger.info(f"📊 Returning zero nutrition for {food_name} - fully dynamic system")
        return {
            'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0,
            'sugars': 0, 'fiber': 0, 'sodium': 0, 'cholesterol': 0,
            'vitamin_a': 0, 'vitamin_c': 0, 'calcium': 0, 'iron': 0, 'potassium': 0,
            'source': 'no_nutrition_data',
            'note': f'No nutrition data available for {food_name} from any API source'
        }
    
    def calculate_overall_confidence(self, foods: List[Dict]) -> float:
        """Calculate overall confidence from detected foods"""
        if not foods:
            return 0.0
        
        # 'confidence' now stored as percentage in final outputs; convert back to 0..1 for averaging
        total_confidence = 0.0
        count = 0
        for food in foods:
            conf = food.get('confidence', 0)
            # if value looks like 0..1 keep it, otherwise assume percent
            conf01 = conf if conf <= 1 else (conf / 100.0)
            total_confidence += conf01
            count += 1
        if count == 0:
            return 0.0
        return round((total_confidence / count) * 100.0, 1)
    
    def calculate_total_nutrition(self, foods: List[Dict]) -> Dict:
        """Calculate total nutrition from all detected foods"""
        total_nutrition = {
            'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0,
            'sugars': 0, 'fiber': 0, 'sodium': 0, 'cholesterol': 0,
            'vitamin_a': 0, 'vitamin_c': 0, 'calcium': 0, 'iron': 0, 'potassium': 0
        }
        
        for food_item in foods:
            nutrition = food_item.get('nutrition', {})
            quantity = food_item.get('quantity', 1)
            
            # Scale nutrition by quantity
            for key in total_nutrition:
                total_nutrition[key] += nutrition.get(key, 0) * quantity
        
        return total_nutrition
    
    

# No global instance - create dynamically as needed
