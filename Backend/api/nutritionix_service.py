#!/usr/bin/env python3
"""
Nutritionix API Service for accurate nutrition data
"""
import requests
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class NutritionixService:
    def __init__(self, app_id: str, api_key: str):
        """
        Initialize Nutritionix Service
        
        Args:
            app_id (str): Nutritionix App ID
            api_key (str): Nutritionix API Key
        """
        self.app_id = app_id
        self.api_key = api_key
        self.base_url = "https://trackapi.nutritionix.com/v2"
        self.headers = {
            "x-app-id": self.app_id,
            "x-app-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def search_food(self, food_name: str) -> Optional[List[Dict]]:
        """
        Search for food using Nutritionix API
        
        Args:
            food_name (str): Name of the food to search for
            
        Returns:
            List of food items found, or None if error
        """
        try:
            url = f"{self.base_url}/search/instant"
            params = {"query": food_name}
            
            logger.info(f"🔍 Searching Nutritionix for: {food_name}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Process common foods
                if 'common' in data and data['common']:
                    for item in data['common'][:5]:  # Limit to top 5
                        results.append({
                            'name': item.get('food_name', ''),
                            'type': 'common',
                            'calories': item.get('nf_calories', 0),
                            'serving_unit': item.get('serving_unit', ''),
                            'serving_qty': item.get('serving_qty', 1),
                            'tag_id': item.get('tag_id', ''),
                            'photo': item.get('photo', {}).get('thumb', '')
                        })
                
                # Process branded foods
                if 'branded' in data and data['branded']:
                    for item in data['branded'][:3]:  # Limit to top 3
                        results.append({
                            'name': item.get('food_name', ''),
                            'type': 'branded',
                            'calories': item.get('nf_calories', 0),
                            'serving_unit': item.get('serving_unit', ''),
                            'serving_qty': item.get('serving_qty', 1),
                            'nix_item_id': item.get('nix_item_id', ''),
                            'brand_name': item.get('brand_name', ''),
                            'photo': item.get('photo', {}).get('thumb', '')
                        })
                
                logger.info(f"✅ Found {len(results)} items for {food_name}")
                return results
                
            else:
                logger.error(f"❌ Nutritionix search failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception in Nutritionix search: {e}")
            return None
    
    def get_detailed_nutrition(self, food_item: Dict) -> Optional[Dict]:
        """
        Get detailed nutrition information for a food item
        
        Args:
            food_item (Dict): Food item from search results
            
        Returns:
            Detailed nutrition data or None if error
        """
        try:
            url = f"{self.base_url}/natural/nutrients"
            
            # Try different query formats
            queries_to_try = []
            
            # Try with nix_item_id first (for branded items)
            if food_item.get('type') == 'branded' and food_item.get('nix_item_id'):
                queries_to_try.append(f"nix_item_id:{food_item['nix_item_id']}")
            
            # Try with tag_id (for common items)
            if food_item.get('tag_id'):
                queries_to_try.append(f"tag_id:{food_item['tag_id']}")
            
            # Try with just the food name
            queries_to_try.append(food_item['name'])
            
            # Try with common food name variations
            food_name = food_item['name'].lower()
            if 'samosa' in food_name:
                queries_to_try.extend(['samosa', 'indian samosa', 'vegetable samosa'])
            elif 'dosa' in food_name:
                queries_to_try.extend(['dosa', 'masala dosa', 'indian dosa'])
            elif 'pizza' in food_name:
                queries_to_try.extend(['pizza', 'pizza slice', 'cheese pizza'])
            
            logger.info(f"🍎 Getting detailed nutrition for: {food_item['name']}")
            
            # Try each query until one works
            for query in queries_to_try:
                try:
                    data = {"query": query}
                    response = requests.post(url, headers=self.headers, json=data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'foods' in result and result['foods']:
                            food_data = result['foods'][0]
                            
                            # Extract comprehensive nutrition data
                            nutrition = {
                                'name': food_data.get('food_name', food_item['name']),
                                'calories': food_data.get('nf_calories', 0),
                                'protein': food_data.get('nf_protein', 0),
                                'carbs': food_data.get('nf_total_carbohydrate', 0),
                                'fats': food_data.get('nf_total_fat', 0),
                                'saturated_fat': food_data.get('nf_saturated_fat', 0),
                                'sugars': food_data.get('nf_sugars', 0),
                                'fiber': food_data.get('nf_dietary_fiber', 0),
                                'sodium': food_data.get('nf_sodium', 0),
                                'cholesterol': food_data.get('nf_cholesterol', 0),
                                'serving_unit': food_data.get('serving_unit', food_item.get('serving_unit', '')),
                                'serving_qty': food_data.get('serving_qty', food_item.get('serving_qty', 1)),
                                'serving_weight_grams': food_data.get('serving_weight_grams', 0),
                                'photo': food_item.get('photo', ''),
                                'source': 'nutritionix'
                            }
                            
                            # Add micronutrients if available
                            nutrition.update({
                                'vitamin_a': food_data.get('nf_vitamin_a_dv', 0),
                                'vitamin_c': food_data.get('nf_vitamin_c_dv', 0),
                                'calcium': food_data.get('nf_calcium_dv', 0),
                                'iron': food_data.get('nf_iron_dv', 0),
                                'potassium': food_data.get('nf_potassium', 0),
                            })
                            
                            logger.info(f"✅ Got detailed nutrition for {nutrition['name']}: {nutrition['calories']} cal")
                            return nutrition
                    else:
                        logger.debug(f"Query '{query}' failed: {response.status_code}")
                        
                except Exception as e:
                    logger.debug(f"Query '{query}' exception: {e}")
                    continue
            
            logger.warning(f"⚠️ No detailed nutrition data found for {food_item['name']} after trying {len(queries_to_try)} queries")
            return None
                
        except Exception as e:
            logger.error(f"❌ Exception getting detailed nutrition: {e}")
            return None
    
    def get_nutrition_for_food(self, food_name: str) -> Optional[Dict]:
        """
        Get nutrition data for a food (searches and gets detailed info)
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            Nutrition data or None if not found
        """
        try:
            # Search for the food
            search_results = self.search_food(food_name)
            
            if not search_results:
                logger.warning(f"⚠️ No search results for {food_name}")
                return None
            
            # Try to get detailed nutrition for the best match
            best_match = search_results[0]  # Use the first (most relevant) result
            
            detailed_nutrition = self.get_detailed_nutrition(best_match)
            
            if detailed_nutrition:
                return detailed_nutrition
            
            # If detailed nutrition fails, return basic info only (no estimates)
            logger.info(f"📋 Returning basic nutrition data for {food_name}")
            
            calories = best_match.get('calories', 0)
            
            return {
                'name': best_match['name'],
                'calories': calories,
                'protein': 0,  # No estimation - fully dynamic
                'carbs': 0,    # No estimation - fully dynamic
                'fats': 0,     # No estimation - fully dynamic
                'saturated_fat': 0,
                'sugars': 0,
                'fiber': 0,
                'sodium': 0,
                'cholesterol': 0,
                'serving_unit': best_match.get('serving_unit', ''),
                'serving_qty': best_match.get('serving_qty', 1),
                'serving_weight_grams': 0,
                'photo': best_match.get('photo', ''),
                'source': 'nutritionix_basic_only',
                'vitamin_a': 0,
                'vitamin_c': 0,
                'calcium': 0,
                'iron': 0,
                'potassium': 0,
                'note': f'Only basic calories available for {food_name}'
            }
            
        except Exception as e:
            logger.error(f"❌ Exception getting nutrition for {food_name}: {e}")
            return None
    
    def test_api(self) -> bool:
        """
        Test if the API is working
        
        Returns:
            True if API is working, False otherwise
        """
        try:
            test_result = self.search_food("apple")
            return test_result is not None and len(test_result) > 0
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False

# Create global instance
nutritionix_service = NutritionixService(
    app_id="46900039",
    api_key="38c1fabb4b7712e09b6d2ddea180c3a3"
)
