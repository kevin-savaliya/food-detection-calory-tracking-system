import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import glob
from typing import List, Dict, Tuple
import logging
import requests
import json
from io import BytesIO

logger = logging.getLogger(__name__)    

class FoodDetector:
    def __init__(self):
        """Initialize the food detector with YOLOv8 model and comprehensive CSV dataset"""
        self.model = None
        self.food_database = {}
        self.load_food_database()
        self.load_model()
    
    def load_food_database(self):
        """Load comprehensive food database from all CSV files in dataset folder"""
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')
            csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
            
            print(f"Loading food database from {len(csv_files)} CSV files...")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    print(f"Loading {csv_file} with {len(df)} food items")
                    
                    for _, row in df.iterrows():
                        food_name = str(row['food']).lower().strip()
                        
                        # Create comprehensive nutrition data
                        nutrition_data = {
                            'calories': float(row.get('Caloric Value', 0)),
                            'protein': float(row.get('Protein', 0)),
                            'carbs': float(row.get('Carbohydrates', 0)),
                            'fats': float(row.get('Fat', 0)),
                            'sugars': float(row.get('Sugars', 0)),
                            'fiber': float(row.get('Dietary Fiber', 0)),
                            'sodium': float(row.get('Sodium', 0)),
                            'cholesterol': float(row.get('Cholesterol', 0)),
                            'vitamin_a': float(row.get('Vitamin A', 0)),
                            'vitamin_c': float(row.get('Vitamin C', 0)),
                            'calcium': float(row.get('Calcium', 0)),
                            'iron': float(row.get('Iron', 0)),
                            'potassium': float(row.get('Potassium', 0)),
                            'serving_size': 100,
                            'serving_unit': 'g'
                        }
                        
                        self.food_database[food_name] = nutrition_data
                        
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
                    continue
            
            print(f"Successfully loaded {len(self.food_database)} food items from dataset")
            
        except Exception as e:
            logger.error(f"Error loading food database: {e}")
            self.food_database = {}
    
    def load_model(self):
        """Load YOLOv8 model for food detection"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("YOLOv8 model loaded successfully")
            else:
                print("YOLOv8 model not found, using basic detection")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_foods_with_yolo(self, image_path: str) -> Tuple[List[str], List[float]]:
        """Detect foods using YOLOv8 model with enhanced mapping"""
        try:
            if self.model is None:
                return [], []
            
            # Run YOLO detection
            results = self.model(image_path)
            
            detected_foods = []
            confidence_scores = []
            
            # Enhanced COCO to food mapping
            food_mapping = {
                # Fruits
                46: 'banana', 47: 'apple', 48: 'orange', 49: 'broccoli', 50: 'carrot',
                51: 'hot dog', 52: 'pizza', 53: 'donut', 54: 'cake', 55: 'chair',
                56: 'couch', 57: 'potted plant', 58: 'dining table', 59: 'toilet',
                60: 'tv', 61: 'laptop', 62: 'mouse', 63: 'remote', 64: 'keyboard',
                65: 'cell phone', 66: 'microwave', 67: 'oven', 68: 'toaster', 69: 'sink',
                70: 'refrigerator', 71: 'book', 72: 'clock', 73: 'vase', 74: 'scissors',
                75: 'teddy bear', 76: 'hair drier', 77: 'toothbrush',
                # Additional food-related items
                78: 'sandwich', 79: 'burger', 80: 'fries', 81: 'salad', 82: 'soup',
                83: 'rice', 84: 'bread', 85: 'cheese', 86: 'meat', 87: 'fish',
                88: 'chicken', 89: 'egg', 90: 'milk', 91: 'yogurt', 92: 'ice cream',
                93: 'cookie', 94: 'chocolate', 95: 'candy', 96: 'chips', 97: 'nuts',
                98: 'vegetables', 99: 'fruits'
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        detected_object = food_mapping.get(class_id, 'unknown')
                        
                        # Only accept food-related detections with high confidence
                        if detected_object in food_mapping.values() and confidence > 0.5:
                            if detected_object not in detected_foods:
                                detected_foods.append(detected_object)
                                confidence_scores.append(confidence)
            
            return detected_foods, confidence_scores
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return [], []
    
    def detect_foods_by_image_analysis(self, image_path: str) -> Dict:
        """Enhanced image analysis for food detection"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Get image dimensions
            height, width = image.shape[:2]
            total_pixels = height * width
            
            # Enhanced segmentation
            segments = self._enhanced_segment_image_components(image, hsv, lab)
            
            detected_foods = []
            confidence_scores = []
            
            # Analyze each segment
            for segment_info in segments:
                analysis = self._enhanced_analyze_segment(segment_info, total_pixels, image)
                detected_foods.extend(analysis['foods'])
                confidence_scores.extend(analysis['confidences'])
            
            # Remove duplicates while preserving order
            unique_foods = []
            unique_confidences = []
            for food, conf in zip(detected_foods, confidence_scores):
                if food not in unique_foods:
                    unique_foods.append(food)
                    unique_confidences.append(conf)
            
            return {
                'detected_foods': unique_foods,
                'confidence_scores': unique_confidences
            }
            
        except Exception as e:
            print(f"Error in image analysis: {str(e)}")
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _enhanced_segment_image_components(self, image, hsv, lab):
        """Enhanced segmentation with more sophisticated color analysis"""
        segments = []
        
        # Enhanced color masks with better ranges
        color_masks = {
            'brown': self._create_enhanced_brown_mask(hsv),
            'red': self._create_enhanced_red_mask(hsv),
            'green': self._create_enhanced_green_mask(hsv),
            'yellow': self._create_enhanced_yellow_mask(hsv),
            'white': self._create_enhanced_white_mask(lab),
            'orange': self._create_enhanced_orange_mask(hsv),
            'pink': self._create_enhanced_pink_mask(hsv),
            'purple': self._create_enhanced_purple_mask(hsv)
        }
        
        # Process each color mask
        for color_name, mask in color_masks.items():
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 200:  # Filter small noise
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate additional features
                        aspect_ratio = w / h if h > 0 else 1
                        solidity = area / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
                        
                        segments.append({
                            'color': color_name,
                            'area': area,
                            'bbox': (x, y, w, h),
                            'contour': contour,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity
                        })
        
        return segments
    
    def _create_enhanced_brown_mask(self, hsv):
        """Create enhanced brown color mask"""
        lower_brown1 = np.array([8, 60, 60])
        upper_brown1 = np.array([25, 255, 255])
        lower_brown2 = np.array([25, 60, 60])
        upper_brown2 = np.array([35, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
        mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_enhanced_red_mask(self, hsv):
        """Create enhanced red color mask"""
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    
    def _create_enhanced_green_mask(self, hsv):
        """Create enhanced green color mask"""
        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])
        return cv2.inRange(hsv, lower_green, upper_green)
    
    def _create_enhanced_yellow_mask(self, hsv):
        """Create enhanced yellow color mask"""
        lower_yellow = np.array([20, 60, 60])
        upper_yellow = np.array([35, 255, 255])
        return cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    def _create_enhanced_white_mask(self, lab):
        """Create enhanced white color mask"""
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([255, 15, 15])
        return cv2.inRange(lab, lower_white, upper_white)
    
    def _create_enhanced_orange_mask(self, hsv):
        """Create enhanced orange color mask"""
        lower_orange = np.array([5, 60, 60])
        upper_orange = np.array([15, 255, 255])
        return cv2.inRange(hsv, lower_orange, upper_orange)
    
    def _create_enhanced_pink_mask(self, hsv):
        """Create enhanced pink color mask"""
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        return cv2.inRange(hsv, lower_pink, upper_pink)
    
    def _create_enhanced_purple_mask(self, hsv):
        """Create enhanced purple color mask"""
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([140, 255, 255])
        return cv2.inRange(hsv, lower_purple, upper_purple)
    
    def _enhanced_analyze_segment(self, segment_info, total_pixels, original_image):
        """Enhanced segment analysis with better food matching"""
        foods = []
        confidences = []
        
        color = segment_info['color']
        area_ratio = segment_info['area'] / total_pixels
        aspect_ratio = segment_info['aspect_ratio']
        solidity = segment_info['solidity']
        
        # Enhanced food matching based on characteristics
        potential_foods = self._enhanced_find_foods_by_characteristics(color, area_ratio, aspect_ratio, solidity)
        
        for food_name, confidence in potential_foods:
            if food_name not in foods:
                foods.append(food_name)
                confidences.append(confidence)
        
        return {'foods': foods, 'confidences': confidences}
    
    def _enhanced_find_foods_by_characteristics(self, color, area_ratio, aspect_ratio, solidity):
        """Enhanced food finding based on multiple characteristics"""
        potential_matches = []
        
        for food_name in self.food_database.keys():
            confidence = 0.0
            
            # Enhanced color-based scoring
            confidence += self._calculate_enhanced_color_score(color, area_ratio, food_name)
            
            # Enhanced size-based scoring
            confidence += self._calculate_enhanced_size_score(area_ratio, food_name)
            
            # Enhanced shape-based scoring
            confidence += self._calculate_enhanced_shape_score(aspect_ratio, solidity, food_name)
            
            # Only include foods with sufficient confidence
            if confidence > 0.3:
                potential_matches.append((food_name, min(confidence, 0.9)))
        
        # Sort by confidence and return top matches
        potential_matches.sort(key=lambda x: x[1], reverse=True)
        return potential_matches[:5]  # Return top 5 matches per segment
    
    def _calculate_enhanced_color_score(self, color, area_ratio, food_name):
        """Enhanced color-based confidence score"""
        score = 0.0
        food_lower = food_name.lower()
        
        # Enhanced color-food mapping
        color_food_mapping = {
            'brown': ['bread', 'toast', 'beef', 'meat', 'chocolate', 'coffee', 'nuts', 'cereal'],
            'red': ['tomato', 'apple', 'cherry', 'strawberry', 'beet', 'pepper', 'meat'],
            'green': ['lettuce', 'spinach', 'grape', 'pear', 'broccoli', 'cucumber', 'peas'],
            'yellow': ['banana', 'lemon', 'cheese', 'corn', 'pepper', 'squash'],
            'orange': ['orange', 'mango', 'carrot', 'pumpkin', 'sweet potato'],
            'white': ['onion', 'rice', 'bread', 'milk', 'yogurt', 'cheese', 'potato'],
            'pink': ['salmon', 'prawn', 'grapefruit', 'watermelon'],
            'purple': ['grape', 'eggplant', 'plum', 'cabbage']
        }
        
        if color in color_food_mapping:
            for food_type in color_food_mapping[color]:
                if food_type in food_lower:
                    score += 0.5
                    break
            else:
                score += 0.1
        
        return score
    
    def _calculate_enhanced_size_score(self, area_ratio, food_name):
        """Enhanced size-based confidence score"""
        score = 0.0
        
        if area_ratio > 0.15:  # Large components
            score += 0.3
        elif area_ratio > 0.08:  # Medium components
            score += 0.2
        elif area_ratio > 0.03:  # Small components
            score += 0.15
        else:  # Very small components
            score += 0.1
        
        return score
    
    def _calculate_enhanced_shape_score(self, aspect_ratio, solidity, food_name):
        """Enhanced shape-based confidence score"""
        score = 0.0
        
        # Aspect ratio scoring
        if 0.8 < aspect_ratio < 1.2:  # Square-ish shapes
            score += 0.2
        elif aspect_ratio > 1.5 or aspect_ratio < 0.7:  # Rectangular shapes
            score += 0.15
        else:  # Other shapes
            score += 0.1
        
        # Solidity scoring (how filled the shape is)
        if solidity > 0.8:  # Solid shapes
            score += 0.1
        elif solidity > 0.6:  # Semi-solid shapes
            score += 0.05
        
        return score
    
    def get_nutrition_from_database(self, food_name: str) -> Dict:
        """Get nutrition information from the loaded CSV database with enhanced matching"""
        try:
            # Try exact match first
            if food_name in self.food_database:
                return self.food_database[food_name]
            
            # Try partial matches with better logic
            best_match = None
            best_score = 0
            
            for db_food_name in self.food_database.keys():
                # Check if food_name is contained in db_food_name or vice versa
                if food_name in db_food_name or db_food_name in food_name:
                    # Calculate similarity score
                    score = len(set(food_name.split()) & set(db_food_name.split())) / max(len(food_name.split()), len(db_food_name.split()))
                    if score > best_score:
                        best_score = score
                        best_match = db_food_name
            
            if best_match and best_score > 0.3:
                return self.food_database[best_match]
            
            # Return empty dict if no match found
            return {}
            
        except Exception as e:
            print(f"Error getting nutrition: {e}")
            return {}
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image and return comprehensive food detection results"""
        try:
            print(f"Analyzing image: {image_path}")
            
            # Try YOLO detection first
            yolo_foods, yolo_confidences = self.detect_foods_with_yolo(image_path)
            
            # Use enhanced image analysis as backup
            analysis_results = self.detect_foods_by_image_analysis(image_path)
            
            if 'error' in analysis_results:
                analysis_foods = []
                analysis_confidences = []
            else:
                analysis_foods = analysis_results.get('detected_foods', [])
                analysis_confidences = analysis_results.get('confidence_scores', [])
            
            # Combine results intelligently
            detected_foods = []
            confidence_scores = []
            
            # Add YOLO detections first (if any)
            for food, conf in zip(yolo_foods, yolo_confidences):
                if food not in detected_foods:
                    detected_foods.append(food)
                    confidence_scores.append(conf)
            
            # Add image analysis detections, avoiding duplicates
            for food, conf in zip(analysis_foods, analysis_confidences):
                if food not in detected_foods:
                    detected_foods.append(food)
                    confidence_scores.append(conf)
            
            # Get nutrition data for detected foods
            nutrition_data = []
            total_nutrition = {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fats': 0,
                'sugars': 0,
                'fiber': 0,
                'sodium': 0,
                'cholesterol': 0,
                'vitamin_a': 0,
                'vitamin_c': 0,
                'calcium': 0,
                'iron': 0,
                'potassium': 0
            }
            
            for food_name in detected_foods:
                nutrition = self.get_nutrition_from_database(food_name)
                if nutrition:  # Only add if nutrition data found
                    nutrition_data.append({
                        'food_name': food_name,
                        'nutrition': nutrition
                    })
                    
                    # Add to total nutrition
                    for key in total_nutrition:
                        if key in nutrition:
                            total_nutrition[key] += nutrition[key]
            
            return {
                'detected_foods': detected_foods,
                'confidence_scores': confidence_scores,
                'nutrition_data': nutrition_data,
                'total_nutrition': total_nutrition
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'detected_foods': [],
                'confidence_scores': [],
                'nutrition_data': [],
                'total_nutrition': {
                    'calories': 0,
                    'protein': 0,
                    'carbs': 0,
                    'fats': 0,
                    'sugars': 0,
                    'fiber': 0,
                    'sodium': 0,
                    'cholesterol': 0,
                    'vitamin_a': 0,
                    'vitamin_c': 0,
                    'calcium': 0,
                    'iron': 0,
                    'potassium': 0
                }
            }
    
    def get_food_suggestions(self, detected_foods: List[str]) -> List[str]:
        """Get dietary suggestions based on detected foods"""
        suggestions = []
        
        if not detected_foods:
            suggestions.append("No specific foods detected. Consider taking a clearer photo.")
            return suggestions
        
        # Analyze nutrition data from detected foods
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fats = 0
        
        for food in detected_foods:
            nutrition = self.get_nutrition_from_database(food)
            total_calories += nutrition.get('calories', 0)
            total_protein += nutrition.get('protein', 0)
            total_carbs += nutrition.get('carbs', 0)
            total_fats += nutrition.get('fats', 0)
        
        # Dynamic suggestions based on nutrition analysis
        if total_protein > 20:
            suggestions.append("Good protein source detected. Consider pairing with vegetables for a balanced meal.")
        
        if total_carbs > 30:
            suggestions.append("Carbohydrate-rich meal detected. Consider portion control for carbohydrate management.")
        
        if total_fats > 15:
            suggestions.append("Higher fat content detected. Consider balancing with lean proteins and vegetables.")
        
        if len(detected_foods) >= 3:
            suggestions.append("Multiple food items detected. This appears to be a balanced meal!")
        
        if total_calories > 500:
            suggestions.append("High-calorie meal detected. Consider portion control and balance with lighter options.")
        
        if len(detected_foods) == 1:
            suggestions.append("Single food item detected. Consider adding variety for better nutritional balance.")
        
        return suggestions

# Create global instance
food_detector = FoodDetector() 