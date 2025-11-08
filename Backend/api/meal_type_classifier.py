"""
AI Calorie Tracking System - Intelligent Meal Type Classifier
Determines meal type based on time of day AND detected food characteristics

Authors: Kevin Savaliya, Nishit Vadhia
Institution: Nirma University, Ahmedabad
"""

import re
from datetime import datetime, time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MealTypeResult:
    meal_type: str
    confidence: float
    reasoning: str
    time_factor: float
    food_factor: float


class IntelligentMealTypeClassifier:
    """Intelligent meal type classifier using time + food characteristics"""
    
    def __init__(self):
        # Time-based meal patterns (24-hour format)
        self.time_patterns = {
            'breakfast': [
                (6, 11),   # 6 AM - 11 AM
                (5, 12),   # Extended breakfast window
            ],
            'lunch': [
                (11, 15),  # 11 AM - 3 PM
                (10, 16),  # Extended lunch window
            ],
            'snack': [
                (15, 18),  # 3 PM - 6 PM
                (14, 19),  # Extended snack window
                (21, 23),  # Late night snack
            ],
            'dinner': [
                (18, 22),  # 6 PM - 10 PM
                (17, 23),  # Extended dinner window
                (0, 6),    # Late night/early morning
            ]
        }
        
        # Food characteristics for each meal type
        self.food_patterns = {
            'breakfast': {
                'primary_foods': [
                    'bread', 'toast', 'cereal', 'oats', 'porridge', 'pancake', 'waffle',
                    'egg', 'omelette', 'scrambled', 'boiled egg', 'fried egg',
                    'milk', 'yogurt', 'curd', 'buttermilk', 'lassi',
                    'fruit', 'banana', 'apple', 'orange', 'berries',
                    'coffee', 'tea', 'juice', 'smoothie',
                    'idli', 'dosa', 'upma', 'poha', 'paratha', 'puri'
                ],
                'secondary_foods': [
                    'butter', 'jam', 'honey', 'sugar', 'cream',
                    'cheese', 'paneer', 'nuts', 'almonds', 'walnuts'
                ],
                'avoid_foods': [
                    'rice', 'dal', 'curry', 'sambar', 'rasam', 'biryani',
                    'pizza', 'burger', 'pasta', 'noodles', 'soup'
                ]
            },
            'lunch': {
                'primary_foods': [
                    'rice', 'dal', 'curry', 'sambar', 'rasam', 'biryani',
                    'roti', 'chapati', 'naan', 'paratha', 'puri',
                    'vegetable', 'sabzi', 'paneer', 'chicken', 'fish', 'mutton',
                    'soup', 'salad', 'raita', 'pickle', 'papad'
                ],
                'secondary_foods': [
                    'butter', 'ghee', 'oil', 'spices', 'herbs',
                    'onion', 'tomato', 'garlic', 'ginger'
                ],
                'avoid_foods': [
                    'cereal', 'oats', 'pancake', 'waffle', 'toast',
                    'coffee', 'tea', 'juice', 'smoothie'
                ]
            },
            'dinner': {
                'primary_foods': [
                    'rice', 'dal', 'curry', 'sambar', 'rasam', 'biryani',
                    'roti', 'chapati', 'naan', 'paratha',
                    'vegetable', 'sabzi', 'paneer', 'chicken', 'fish', 'mutton',
                    'soup', 'salad', 'raita', 'pickle', 'papad',
                    'pasta', 'noodles', 'pizza', 'burger'
                ],
                'secondary_foods': [
                    'butter', 'ghee', 'oil', 'spices', 'herbs',
                    'onion', 'tomato', 'garlic', 'ginger', 'cheese'
                ],
                'avoid_foods': [
                    'cereal', 'oats', 'pancake', 'waffle', 'toast',
                    'coffee', 'tea', 'juice', 'smoothie'
                ]
            },
            'snack': {
                'primary_foods': [
                    'samosa', 'pakora', 'vada', 'bonda', 'cutlet',
                    'biscuit', 'cookie', 'cake', 'pastry', 'muffin',
                    'chips', 'namkeen', 'mixture', 'sev', 'chivda',
                    'fruit', 'banana', 'apple', 'orange', 'grapes',
                    'nuts', 'almonds', 'walnuts', 'cashews', 'peanuts',
                    'tea', 'coffee', 'juice', 'smoothie', 'lassi',
                    'pav bhaji', 'vada pav', 'dosa', 'idli', 'medu vada'
                ],
                'secondary_foods': [
                    'chutney', 'sauce', 'dip', 'spread',
                    'sugar', 'honey', 'jam', 'butter'
                ],
                'avoid_foods': [
                    'rice', 'dal', 'curry', 'sambar', 'rasam',
                    'roti', 'chapati', 'naan', 'paratha'
                ]
            }
        }
        
        # Meal type weights
        self.weights = {
            'time': 0.4,      # 40% weight to time of day
            'food': 0.6       # 60% weight to food characteristics
        }
    
    def classify_meal_type(self, detected_foods: List[Dict], 
                          current_time: datetime = None) -> MealTypeResult:
        """
        Classify meal type based on time and detected foods
        
        Args:
            detected_foods: List of detected food items with names
            current_time: Current datetime (defaults to now)
            
        Returns:
            MealTypeResult with meal type, confidence, and reasoning
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Extract food names
        food_names = [food.get('name', '').lower() for food in detected_foods]
        
        # Calculate time-based score
        time_scores = self._calculate_time_scores(current_time)
        
        # Calculate food-based scores
        food_scores = self._calculate_food_scores(food_names)
        
        # Combine scores with weights
        final_scores = {}
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
            final_scores[meal_type] = (
                self.weights['time'] * time_scores[meal_type] +
                self.weights['food'] * food_scores[meal_type]
            )
        
        # Find best match
        best_meal_type = max(final_scores, key=final_scores.get)
        best_confidence = final_scores[best_meal_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_meal_type, time_scores, food_scores, food_names, current_time
        )
        
        return MealTypeResult(
            meal_type=best_meal_type,
            confidence=best_confidence,
            reasoning=reasoning,
            time_factor=time_scores[best_meal_type],
            food_factor=food_scores[best_meal_type]
        )
    
    def _calculate_time_scores(self, current_time: datetime) -> Dict[str, float]:
        """Calculate time-based scores for each meal type"""
        current_hour = current_time.hour
        scores = {'breakfast': 0.0, 'lunch': 0.0, 'dinner': 0.0, 'snack': 0.0}
        
        for meal_type, time_ranges in self.time_patterns.items():
            max_score = 0.0
            for start_hour, end_hour in time_ranges:
                if start_hour <= end_hour:
                    # Normal time range (e.g., 6-11)
                    if start_hour <= current_hour < end_hour:
                        # Calculate proximity to center of time range
                        center = (start_hour + end_hour) / 2
                        distance = abs(current_hour - center)
                        max_range = (end_hour - start_hour) / 2
                        score = max(0, 1 - (distance / max_range))
                        max_score = max(max_score, score)
                else:
                    # Overnight range (e.g., 0-6)
                    if current_hour >= start_hour or current_hour < end_hour:
                        # Calculate proximity to center
                        if current_hour >= start_hour:
                            center = (start_hour + 24 + end_hour) / 2
                            if center > 24:
                                center -= 24
                            distance = abs(current_hour - center)
                        else:
                            center = (start_hour + end_hour) / 2
                            distance = abs(current_hour + 24 - center)
                        max_range = (24 - start_hour + end_hour) / 2
                        score = max(0, 1 - (distance / max_range))
                        max_score = max(max_score, score)
            
            scores[meal_type] = max_score
        
        return scores
    
    def _calculate_food_scores(self, food_names: List[str]) -> Dict[str, float]:
        """Calculate food-based scores for each meal type"""
        scores = {'breakfast': 0.0, 'lunch': 0.0, 'dinner': 0.0, 'snack': 0.0}
        
        for meal_type, patterns in self.food_patterns.items():
            score = 0.0
            total_foods = len(food_names)
            
            if total_foods == 0:
                scores[meal_type] = 0.0
                continue
            
            for food_name in food_names:
                food_lower = food_name.lower().strip()
                
                # Check primary foods (high weight)
                for primary_food in patterns['primary_foods']:
                    if self._food_matches(food_lower, primary_food):
                        score += 1.0
                        break
                else:
                    # Check secondary foods (medium weight)
                    for secondary_food in patterns['secondary_foods']:
                        if self._food_matches(food_lower, secondary_food):
                            score += 0.5
                            break
                    else:
                        # Check avoid foods (negative weight)
                        for avoid_food in patterns['avoid_foods']:
                            if self._food_matches(food_lower, avoid_food):
                                score -= 0.3
                                break
            
            # Normalize score
            scores[meal_type] = max(0, min(1, score / total_foods))
        
        return scores
    
    def _food_matches(self, food_name: str, pattern: str) -> bool:
        """Check if food name matches pattern (with fuzzy matching)"""
        food_name = food_name.lower().strip()
        pattern = pattern.lower().strip()
        
        # Exact match
        if food_name == pattern:
            return True
        
        # Contains match
        if pattern in food_name or food_name in pattern:
            return True
        
        # Word-based matching
        food_words = set(food_name.split())
        pattern_words = set(pattern.split())
        
        if pattern_words.intersection(food_words):
            return True
        
        return False
    
    def _generate_reasoning(self, meal_type: str, time_scores: Dict[str, float], 
                          food_scores: Dict[str, float], food_names: List[str], 
                          current_time: datetime) -> str:
        """Generate human-readable reasoning for the classification"""
        current_hour = current_time.hour
        time_str = current_time.strftime("%I:%M %p")
        
        reasoning_parts = []
        
        # Time reasoning
        time_score = time_scores[meal_type]
        if time_score > 0.7:
            reasoning_parts.append(f"Time ({time_str}) strongly suggests {meal_type}")
        elif time_score > 0.4:
            reasoning_parts.append(f"Time ({time_str}) moderately suggests {meal_type}")
        else:
            reasoning_parts.append(f"Time ({time_str}) weakly suggests {meal_type}")
        
        # Food reasoning
        food_score = food_scores[meal_type]
        if food_score > 0.7:
            reasoning_parts.append(f"Detected foods strongly match {meal_type} patterns")
        elif food_score > 0.4:
            reasoning_parts.append(f"Detected foods moderately match {meal_type} patterns")
        else:
            reasoning_parts.append(f"Detected foods weakly match {meal_type} patterns")
        
        # Specific food mentions
        if food_names:
            primary_matches = []
            for food_name in food_names:
                for primary_food in self.food_patterns[meal_type]['primary_foods']:
                    if self._food_matches(food_name, primary_food):
                        primary_matches.append(food_name)
                        break
            
            if primary_matches:
                reasoning_parts.append(f"Key {meal_type} foods detected: {', '.join(primary_matches[:3])}")
        
        return ". ".join(reasoning_parts) + "."


# Example usage and testing
def test_meal_classifier():
    """Test the meal type classifier with various scenarios"""
    classifier = IntelligentMealTypeClassifier()
    
    test_cases = [
        {
            'name': 'Morning Breakfast',
            'foods': [{'name': 'bread'}, {'name': 'egg'}, {'name': 'milk'}],
            'time': datetime.now().replace(hour=8, minute=30)
        },
        {
            'name': 'Lunch Time',
            'foods': [{'name': 'rice'}, {'name': 'dal'}, {'name': 'curry'}],
            'time': datetime.now().replace(hour=13, minute=0)
        },
        {
            'name': 'Evening Snack',
            'foods': [{'name': 'samosa'}, {'name': 'tea'}],
            'time': datetime.now().replace(hour=16, minute=30)
        },
        {
            'name': 'Dinner Time',
            'foods': [{'name': 'roti'}, {'name': 'paneer'}, {'name': 'rice'}],
            'time': datetime.now().replace(hour=19, minute=30)
        },
        {
            'name': 'Late Night Snack',
            'foods': [{'name': 'biscuit'}, {'name': 'coffee'}],
            'time': datetime.now().replace(hour=22, minute=0)
        }
    ]
    
    print("=" * 80)
    print("MEAL TYPE CLASSIFIER TEST RESULTS")
    print("=" * 80)
    
    for test_case in test_cases:
        result = classifier.classify_meal_type(
            test_case['foods'], 
            test_case['time']
        )
        
        print(f"\n{test_case['name']}:")
        print(f"  Time: {test_case['time'].strftime('%I:%M %p')}")
        print(f"  Foods: {[f['name'] for f in test_case['foods']]}")
        print(f"  -> Meal Type: {result.meal_type.upper()}")
        print(f"  -> Confidence: {result.confidence:.1%}")
        print(f"  -> Time Factor: {result.time_factor:.1%}")
        print(f"  -> Food Factor: {result.food_factor:.1%}")
        print(f"  -> Reasoning: {result.reasoning}")


if __name__ == "__main__":
    test_meal_classifier()
