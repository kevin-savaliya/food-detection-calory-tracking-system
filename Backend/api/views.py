import os
import logging
from datetime import datetime, timedelta
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.utils import timezone
from django.conf import settings
from django.db.models import Sum, Q, Avg, Count
from django.db.models.functions import TruncDate
from datetime import datetime, timedelta
import calendar
from rest_framework import status, permissions, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token
from .models import UserProfile, FoodItem, FoodDetectionLog, DailyNutritionLog, NutritionGoal
from .serializers import (
    UserRegistrationSerializer, UserLoginSerializer, UserSerializer,
    UserProfileSerializer, FoodItemSerializer, FoodDetectionLogSerializer,
    DailyNutritionLogSerializer, NutritionGoalSerializer
)
from .logmeal_food_detector import LogMealFoodDetector

logger = logging.getLogger(__name__)

# Test endpoint
@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def test_api(request):
    return Response({
        'message': 'AI Calorie Tracking API is working!',
        'status': 'success',
        'timestamp': timezone.now().isoformat()
    })

# Welcome endpoint
@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def welcome_api(request):
    return Response({
        'message': '🎉 Welcome to AI Calorie Tracking System!',
        'status': 'Backend Server Running Successfully',
        'version': '1.0.0',
        'features': [
            'AI-powered food detection',
            'Personalized nutrition tracking',
            'Real-time progress monitoring',
            'Smart dietary suggestions'
        ],
        'endpoints': {
            'test': '/api/test/',
            'welcome': '/api/welcome/',
            'register': '/api/auth/register/',
            'login': '/api/auth/login/',
            'profile': '/api/profile/',
            'food-detection': '/api/detect-food/',
            'nutrition-log': '/api/nutrition-log/',
            'nutrition-summary': '/api/nutrition-summary/',
            'weekly-report': '/api/weekly-report/'
        },
        'timestamp': timezone.now().isoformat(),
        'server_status': '🟢 Online'
    })

class UserRegistrationView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        try:
            serializer = UserRegistrationSerializer(data=request.data)
            
            if serializer.is_valid():
                user = serializer.save()
                token, created = Token.objects.get_or_create(user=user)
                
                # Get user profile data
                try:
                    profile = UserProfile.objects.get(user=user)
                    profile_data = UserProfileSerializer(profile).data
                except UserProfile.DoesNotExist:
                    profile_data = None
                
                return Response({
                    'token': token.key,
                    'user': UserSerializer(user).data,
                    'profile': profile_data,
                    'message': 'User registered successfully'
                }, status=status.HTTP_201_CREATED)
            else:
                # Format error messages for better user experience
                formatted_errors = {}
                for field, errors in serializer.errors.items():
                    if field == 'username':
                        if 'unique' in str(errors).lower():
                            formatted_errors[field] = ['This username is already taken. Please choose a different one.']
                        else:
                            formatted_errors[field] = errors
                    elif field == 'email':
                        if 'unique' in str(errors).lower():
                            formatted_errors[field] = ['This email is already registered. Please use a different email or try logging in.']
                        else:
                            formatted_errors[field] = errors
                    elif field == 'password':
                        formatted_errors[field] = ['Password must be at least 8 characters long and cannot be too common.']
                    elif field == 'password_confirm':
                        formatted_errors[field] = ['Passwords do not match. Please make sure both passwords are identical.']
                    elif field == 'non_field_errors':
                        formatted_errors[field] = errors
                    else:
                        formatted_errors[field] = errors
                
                return Response(formatted_errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Log the error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Registration error: {str(e)}")
            
            # Return a user-friendly error message
            return Response({
                'error': 'Registration failed. Please check your input and try again.',
                'details': 'There was an error processing your registration. Please ensure all fields are filled correctly.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserLoginView(APIView):
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if username and password:
            user = authenticate(username=username, password=password)
            if user:
                token, created = Token.objects.get_or_create(user=user)
                return Response({
                    'token': token.key,
                    'user': UserSerializer(user).data,
                    'message': 'Login successful'
                })
            else:
                return Response({
                    'error': 'Invalid credentials'
                }, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({
                'error': 'Username and password are required'
            }, status=status.HTTP_400_BAD_REQUEST)

class UserProfileView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        try:
            profile = UserProfile.objects.get(user=request.user)
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data)
        except UserProfile.DoesNotExist:
            return Response({
                'error': 'Profile not found'
            }, status=status.HTTP_404_NOT_FOUND)
    
    def put(self, request):
        try:
            profile = UserProfile.objects.get(user=request.user)
            serializer = UserProfileSerializer(profile, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except UserProfile.DoesNotExist:
            return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)

class FoodDetectionView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            # Debug: Print request information
            print(f"Request FILES: {request.FILES}")
            print(f"Request DATA: {request.data}")
            print(f"Request content type: {request.content_type}")
            
            # Check if image file is provided
            if 'image' not in request.FILES:
                print("No image file found in request.FILES")
                return Response({
                    'error': 'No image file provided. Please upload an image.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            image = request.FILES['image']
            print(f"Image file: {image.name}, size: {image.size}, type: {image.content_type}")
            
            # Validate file type
            if not image.content_type.startswith('image/'):
                return Response({
                    'error': 'Invalid file type. Please upload an image file (JPEG, PNG, etc.).'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                return Response({
                    'error': 'File size too large. Please upload an image smaller than 10MB.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Create media directory if it doesn't exist
            media_dir = os.path.join('media', 'food_detection')
            os.makedirs(media_dir, exist_ok=True)
            
            # Save image with unique filename
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{request.user.id}_{timestamp}_{image.name}"
            image_path = os.path.join(media_dir, filename)
            
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            # Use LogMeal Food AI for food detection
            analysis_result = None
            detection_method = "logmeal_api"
            
            try:
                # Initialize LogMeal detector (you'll need to provide your API key)
                logmeal_api_key = "a1e9d93ff53d53676b6ac8e911b1bd5451c46a5c"  # Replace with your actual LogMeal API key
                # logmeal_api_key = "08245cb1bfec31cebf83f0754808b5e33648f0fa"  # Replace with your actual LogMeal API key
                logmeal_detector = LogMealFoodDetector(logmeal_api_key)
                
                print(f"🔍 DEBUG: Processing image: {image_path}")
                print(f"🔍 DEBUG: Image file exists: {os.path.exists(image_path)}")
                print(f"🔍 DEBUG: Image file size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} bytes")
                
                # Use LogMeal Food AI for food detection
                analysis_result = logmeal_detector.detect_foods_logmeal(image_path)
                
                # Check for errors in analysis
                if 'error' not in analysis_result and 'detected_foods' in analysis_result:
                    detection_method = "logmeal_api"
                    print("✅ Food detection successful")
                    
                    # The analysis result is already in the correct format from LogMeal
                    detected_foods_list = analysis_result.get('detected_foods', [])
                    print(f"Detected {len(detected_foods_list)} food items via LogMeal Food AI")
                    
                    analysis_result['detection_method'] = 'logmeal_api'
                else:
                    error_msg = analysis_result.get('error', 'Unknown error')
                    print(f"⚠️ Food detection failed: {error_msg}")
                    raise Exception(f"Food detection failed: {error_msg}")
                    
            except Exception as logmeal_error:
                print(f"❌ Food detection failed: {logmeal_error}")
                return Response({
                    'error': 'Failed to detect food.',
                    'details': str(logmeal_error)
                }, status=500)
            
            # Extract detected foods and confidence scores for logging
            detected_foods = [food['name'] for food in analysis_result.get('detected_foods', [])]
            confidence_scores = [food['confidence'] for food in analysis_result.get('detected_foods', [])]
            
            # Save detection log
            detection_log = FoodDetectionLog.objects.create(
                user=request.user,
                image=image,
                detected_foods=detected_foods,
                confidence_scores=confidence_scores
            )
            
            # Save detected foods to daily nutrition log with enhanced tracking
            saved_foods = []
            detected_foods_list = analysis_result.get('detected_foods', [])
            
            if detected_foods_list:
                today = timezone.now().date()
                
                # Get user profile and determine meal type intelligently
                try:
                    user_profile = UserProfile.objects.get(user=request.user)
                    
                    # Use intelligent meal type classifier
                    from .meal_type_classifier import IntelligentMealTypeClassifier
                    classifier = IntelligentMealTypeClassifier()
                    meal_result = classifier.classify_meal_type(detected_foods_list)
                    meal_type = meal_result.meal_type
                    
                    # Log the intelligent classification
                    print(f" Intelligent Meal Classification:")
                    print(f"   Time: {timezone.now().strftime('%I:%M %p')}")
                    print(f"   Foods: {[f['name'] for f in detected_foods_list]}")
                    print(f"   → Meal Type: {meal_type.upper()}")
                    print(f"   → Confidence: {meal_result.confidence:.1%}")
                    print(f"   → Reasoning: {meal_result.reasoning}")
                    
                except UserProfile.DoesNotExist:
                    # Fallback to simple time-based classification
                    current_hour = timezone.now().hour
                    if 6 <= current_hour < 11:
                        meal_type = 'breakfast'
                    elif 11 <= current_hour < 15:
                        meal_type = 'lunch'
                    elif 15 <= current_hour < 18:
                        meal_type = 'snack'
                    else:
                        meal_type = 'dinner'
                
                for food_item in detected_foods_list:
                    try:
                        food_name = food_item['name']
                        nutrition = food_item.get('nutrition', {})
                        # Normalize confidence to percentage string for notes
                        conf_val = food_item.get('confidence', 0)
                        conf_display = f"{conf_val:.1f}%" if conf_val > 1 else f"{conf_val * 100:.1f}%"
                        
                        # Create daily nutrition log entry
                        daily_log = DailyNutritionLog.objects.create(
                            user=request.user,
                            date=today,
                            meal_type=meal_type,
                            custom_food_name=food_name,
                            quantity=1,  # Default quantity
                            calories=nutrition.get('calories', 0),
                            protein=nutrition.get('protein', 0),
                            carbs=nutrition.get('carbs', 0),
                            fats=nutrition.get('fats', 0),
                            notes=f"Detected via AI with {conf_display} confidence"
                        )
                        
                        saved_foods.append({
                            'food_name': food_name,
                            'calories': nutrition.get('calories', 0),
                            'protein': nutrition.get('protein', 0),
                            'carbs': nutrition.get('carbs', 0),
                            'fats': nutrition.get('fats', 0),
                            'confidence': food_item.get('confidence', 0),
                            'meal_type': meal_type,
                            'log_id': daily_log.id
                        })
                        
                    except Exception as e:
                        logger.error(f"Error saving food item {food_item.get('name', 'unknown')}: {e}")
                        continue
            
            # Clean up temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Get nutrition summary from analysis result
            nutrition_summary = analysis_result.get('nutrition_summary', {})
            
            # Format response with better structure
            response_data = {
                'success': True,
                'detection_log_id': detection_log.id,
                'detected_foods': detected_foods,
                'confidence_scores': [round(score if score > 1 else score * 100, 1) for score in confidence_scores],
                'detailed_analysis': analysis_result.get('detected_foods', []),
                'nutrition_summary': nutrition_summary,
                'image_description': analysis_result.get('image_description', ''),
                'overall_confidence': analysis_result.get('overall_confidence', 0),
                'saved_foods': saved_foods,
                'message': f"Successfully detected {len(detected_foods)} food items and saved to daily log"
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in food detection: {e}")
            return Response({
                'error': 'Error processing image. Please try again with a different image.',
                'details': str(e) if settings.DEBUG else None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FoodItemListView(generics.ListCreateAPIView):
    queryset = FoodItem.objects.all()
    serializer_class = FoodItemSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        queryset = FoodItem.objects.all()
        name = self.request.query_params.get('name', None)
        if name:
            queryset = queryset.filter(name__icontains=name)
        return queryset

class DailyNutritionLogView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        date = request.query_params.get('date', timezone.now().date())
        if isinstance(date, str):
            try:
                date = datetime.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                return Response({
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        logs = DailyNutritionLog.objects.filter(user=request.user, date=date)
        serializer = DailyNutritionLogSerializer(logs, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = DailyNutritionLogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class NutritionSummaryView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        date = request.query_params.get('date', timezone.now().date())
        if isinstance(date, str):
            try:
                date = datetime.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                return Response({
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get user's nutrition goals
        try:
            profile = UserProfile.objects.get(user=request.user)
            target_calories = profile.target_calories
            target_protein = profile.target_protein
            target_carbs = profile.target_carbs
            target_fats = profile.target_fats
        except UserProfile.DoesNotExist:
            target_calories = 2000
            target_protein = 150
            target_carbs = 250
            target_fats = 65
        
        # Calculate actual nutrition for the day
        daily_logs = DailyNutritionLog.objects.filter(user=request.user, date=date)
        
        total_calories = daily_logs.aggregate(Sum('calories'))['calories__sum'] or 0
        total_protein = daily_logs.aggregate(Sum('protein'))['protein__sum'] or 0
        total_carbs = daily_logs.aggregate(Sum('carbs'))['carbs__sum'] or 0
        total_fats = daily_logs.aggregate(Sum('fats'))['fats__sum'] or 0
        
        # Calculate progress percentages
        calories_progress = round((total_calories / target_calories) * 100, 1) if target_calories > 0 else 0
        protein_progress = round((float(total_protein) / target_protein) * 100, 1) if target_protein > 0 else 0
        carbs_progress = round((float(total_carbs) / target_carbs) * 100, 1) if target_carbs > 0 else 0
        fats_progress = round((float(total_fats) / target_fats) * 100, 1) if target_fats > 0 else 0
        
        # Get meal breakdown
        meal_breakdown = {}
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
            meal_logs = daily_logs.filter(meal_type=meal_type)
            meal_calories = meal_logs.aggregate(Sum('calories'))['calories__sum'] or 0
            meal_breakdown[meal_type] = meal_calories
        
        summary_data = {
            'date': date,
            'total_calories': total_calories,
            'total_protein': total_protein,
            'total_carbs': total_carbs,
            'total_fats': total_fats,
            'target_calories': target_calories,
            'target_protein': target_protein,
            'target_carbs': target_carbs,
            'target_fats': target_fats,
            'calories_progress': calories_progress,
            'protein_progress': protein_progress,
            'carbs_progress': carbs_progress,
            'fats_progress': fats_progress,
            'meal_breakdown': meal_breakdown
        }
        
        serializer = NutritionSummarySerializer(summary_data)
        return Response(serializer.data)

class WeeklyReportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        end_date = request.query_params.get('end_date', timezone.now().date())
        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            except ValueError:
                return Response({
                    'error': 'Invalid date format. Use YYYY-MM-DD'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        start_date = end_date - timedelta(days=6)
        
        # Get weekly nutrition data
        weekly_logs = DailyNutritionLog.objects.filter(
            user=request.user,
            date__range=[start_date, end_date]
        )
        
        weekly_data = []
        for i in range(7):
            current_date = start_date + timedelta(days=i)
            day_logs = weekly_logs.filter(date=current_date)
            
            total_calories = day_logs.aggregate(Sum('calories'))['calories__sum'] or 0
            total_protein = day_logs.aggregate(Sum('protein'))['protein__sum'] or 0
            total_carbs = day_logs.aggregate(Sum('carbs'))['carbs__sum'] or 0
            total_fats = day_logs.aggregate(Sum('fats'))['fats__sum'] or 0
            
            weekly_data.append({
                'date': current_date,
                'calories': total_calories,
                'protein': total_protein,
                'carbs': total_carbs,
                'fats': total_fats
            })
        
        return Response({
            'start_date': start_date,
            'end_date': end_date,
            'weekly_data': weekly_data
        })

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def user_logout(request):
    try:
        request.user.auth_token.delete()
        return Response({'message': 'Logged out successfully'})
    except:
        return Response({'error': 'Error logging out'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def detection_history(request):
    """Get user's food detection history"""
    try:
        # Get user's detection history, ordered by most recent first
        detections = FoodDetectionLog.objects.filter(
            user=request.user
        ).order_by('-created_at')[:10]  # Get last 10 detections
        
        detection_data = []
        for detection in detections:
            detection_data.append({
                'id': detection.id,
                'detected_foods': detection.detected_foods,
                'confidence_scores': detection.confidence_scores,
                'created_at': detection.created_at.isoformat(),
                'image_url': detection.image.url if detection.image else None
            })
        
        return Response({
            'success': True,
            'detections': detection_data,
            'count': len(detection_data)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error fetching detection history: {e}")
        return Response({
            'error': 'Failed to fetch detection history'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ==================== ENHANCED NUTRITION TRACKING ====================

class DashboardView(APIView):
    """Comprehensive dashboard with nutrition progress and insights"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        try:
            today = timezone.now().date()
            
            # Get user profile
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                target_calories = user_profile.target_calories
                target_protein = user_profile.target_protein
                target_carbs = user_profile.target_carbs
                target_fats = user_profile.target_fats
            except UserProfile.DoesNotExist:
                target_calories = target_protein = target_carbs = target_fats = 0
            
            # Get today's nutrition totals
            today_nutrition = DailyNutritionLog.objects.filter(
                user=request.user,
                date=today
            ).aggregate(
                total_calories=Sum('calories'),
                total_protein=Sum('protein'),
                total_carbs=Sum('carbs'),
                total_fats=Sum('fats')
            )
            
            # Calculate progress percentages
            calories_progress = (today_nutrition['total_calories'] or 0) / target_calories * 100 if target_calories > 0 else 0
            protein_progress = (today_nutrition['total_protein'] or 0) / target_protein * 100 if target_protein > 0 else 0
            carbs_progress = (today_nutrition['total_carbs'] or 0) / target_carbs * 100 if target_carbs > 0 else 0
            fats_progress = (today_nutrition['total_fats'] or 0) / target_fats * 100 if target_fats > 0 else 0
            
            # Get recent meals
            recent_meals = DailyNutritionLog.objects.filter(
                user=request.user,
                date=today
            ).order_by('-created_at')[:5]
            
            meals_data = []
            for meal in recent_meals:
                meals_data.append({
                    'id': meal.id,
                    'food_name': meal.custom_food_name,
                    'meal_type': meal.meal_type,
                    'calories': meal.calories,
                    'protein': meal.protein,
                    'carbs': meal.carbs,
                    'fats': meal.fats,
                    'quantity': meal.quantity,
                    'created_at': meal.created_at.isoformat()
                })
            
            # Get weekly progress (last 7 days)
            week_start = today - timedelta(days=6)
            weekly_nutrition = DailyNutritionLog.objects.filter(
                user=request.user,
                date__gte=week_start,
                date__lte=today
            ).values('date').annotate(
                daily_calories=Sum('calories'),
                daily_protein=Sum('protein'),
                daily_carbs=Sum('carbs'),
                daily_fats=Sum('fats')
            ).order_by('date')
            
            weekly_data = []
            for day_data in weekly_nutrition:
                weekly_data.append({
                    'date': day_data['date'].isoformat(),
                    'calories': day_data['daily_calories'] or 0,
                    'protein': day_data['daily_protein'] or 0,
                    'carbs': day_data['daily_carbs'] or 0,
                    'fats': day_data['daily_fats'] or 0
                })
            
            # Calculate streaks and insights
            consecutive_days = self.calculate_tracking_streak(request.user)
            
            return Response({
                'success': True,
                'dashboard_data': {
                    'today': {
                        'date': today.isoformat(),
                        'nutrition': {
                            'calories': today_nutrition['total_calories'] or 0,
                            'protein': today_nutrition['total_protein'] or 0,
                            'carbs': today_nutrition['total_carbs'] or 0,
                            'fats': today_nutrition['total_fats'] or 0
                        },
                        'targets': {
                            'calories': target_calories,
                            'protein': target_protein,
                            'carbs': target_carbs,
                            'fats': target_fats
                        },
                        'progress': {
                            'calories': round(calories_progress, 1),
                            'protein': round(protein_progress, 1),
                            'carbs': round(carbs_progress, 1),
                            'fats': round(fats_progress, 1)
                        }
                    },
                    'recent_meals': meals_data,
                    'weekly_progress': weekly_data,
                    'insights': {
                        'tracking_streak': consecutive_days,
                        'calories_remaining': max(0, target_calories - (today_nutrition['total_calories'] or 0)),
                        'protein_remaining': max(0, target_protein - (today_nutrition['total_protein'] or 0)),
                        'carbs_remaining': max(0, target_carbs - (today_nutrition['total_carbs'] or 0)),
                        'fats_remaining': max(0, target_fats - (today_nutrition['total_fats'] or 0))
                    }
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching dashboard data: {e}")
            return Response({
                'error': 'Failed to fetch dashboard data'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def calculate_tracking_streak(self, user):
        """Calculate consecutive days of nutrition tracking"""
        today = timezone.now().date()
        streak = 0
        
        for i in range(30):  # Check last 30 days
            check_date = today - timedelta(days=i)
            has_entries = DailyNutritionLog.objects.filter(
                user=user,
                date=check_date
            ).exists()
            
            if has_entries:
                streak += 1
            else:
                break
        
        return streak

class EnhancedNutritionLogView(APIView):
    """Enhanced nutrition log with meal management and history"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get nutrition log for a specific date or today"""
        try:
            date_param = request.GET.get('date')
            if date_param:
                try:
                    target_date = datetime.strptime(date_param, '%Y-%m-%d').date()
                except ValueError:
                    target_date = timezone.now().date()
            else:
                target_date = timezone.now().date()
            
            # Get nutrition entries for the date
            nutrition_entries = DailyNutritionLog.objects.filter(
                user=request.user,
                date=target_date
            ).order_by('meal_type', '-created_at')
            
            # Group by meal type
            meals = {
                'breakfast': [],
                'lunch': [],
                'dinner': [],
                'snack': []
            }
            
            total_nutrition = {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fats': 0
            }
            
            for entry in nutrition_entries:
                meal_data = {
                    'id': entry.id,
                    'food_name': entry.custom_food_name,
                    'quantity': entry.quantity,
                    'calories': entry.calories,
                    'protein': entry.protein,
                    'carbs': entry.carbs,
                    'fats': entry.fats,
                    'notes': entry.notes,
                    'created_at': entry.created_at.isoformat()
                }
                
                meals[entry.meal_type].append(meal_data)
                
                # Add to totals
                total_nutrition['calories'] += entry.calories
                total_nutrition['protein'] += entry.protein
                total_nutrition['carbs'] += entry.carbs
                total_nutrition['fats'] += entry.fats
            
            # Get user targets
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                targets = {
                    'calories': user_profile.target_calories,
                    'protein': user_profile.target_protein,
                    'carbs': user_profile.target_carbs,
                    'fats': user_profile.target_fats
                }
            except UserProfile.DoesNotExist:
                targets = {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
            
            return Response({
                'success': True,
                'date': target_date.isoformat(),
                'meals': meals,
                'total_nutrition': total_nutrition,
                'targets': targets,
                'progress': {
                    'calories': round((total_nutrition['calories'] / targets['calories'] * 100) if targets['calories'] > 0 else 0, 1),
                    'protein': round((total_nutrition['protein'] / targets['protein'] * 100) if targets['protein'] > 0 else 0, 1),
                    'carbs': round((total_nutrition['carbs'] / targets['carbs'] * 100) if targets['carbs'] > 0 else 0, 1),
                    'fats': round((total_nutrition['fats'] / targets['fats'] * 100) if targets['fats'] > 0 else 0, 1)
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error fetching nutrition log: {e}")
            return Response({
                'error': 'Failed to fetch nutrition log'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request):
        """Add manual nutrition entry"""
        try:
            data = request.data
            
            # Create manual nutrition entry
            nutrition_entry = DailyNutritionLog.objects.create(
                user=request.user,
                date=data.get('date', timezone.now().date()),
                meal_type=data.get('meal_type', 'snack'),
                custom_food_name=data.get('food_name', 'Manual Entry'),
                quantity=data.get('quantity', 1),
                calories=data.get('calories', 0),
                protein=data.get('protein', 0),
                carbs=data.get('carbs', 0),
                fats=data.get('fats', 0),
                notes=data.get('notes', 'Manual entry')
            )
            
            return Response({
                'success': True,
                'message': 'Nutrition entry added successfully',
                'entry': {
                    'id': nutrition_entry.id,
                    'food_name': nutrition_entry.custom_food_name,
                    'meal_type': nutrition_entry.meal_type,
                    'calories': nutrition_entry.calories,
                    'protein': nutrition_entry.protein,
                    'carbs': nutrition_entry.carbs,
                    'fats': nutrition_entry.fats
                }
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error adding nutrition entry: {e}")
            return Response({
                'error': 'Failed to add nutrition entry'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request):
        """Delete nutrition entry"""
        try:
            entry_id = request.data.get('entry_id')
            if not entry_id:
                return Response({
                    'error': 'Entry ID is required'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                entry = DailyNutritionLog.objects.get(
                    id=entry_id,
                    user=request.user
                )
                entry.delete()
                
                return Response({
                    'success': True,
                    'message': 'Nutrition entry deleted successfully'
                }, status=status.HTTP_200_OK)
                
            except DailyNutritionLog.DoesNotExist:
                return Response({
                    'error': 'Nutrition entry not found'
                }, status=status.HTTP_404_NOT_FOUND)
                
        except Exception as e:
            logger.error(f"Error deleting nutrition entry: {e}")
            return Response({
                'error': 'Failed to delete nutrition entry'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class NutritionReportsView(APIView):
    """Comprehensive nutrition reports and analytics"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        """Get nutrition reports for different time periods"""
        try:
            report_type = request.GET.get('type', 'weekly')  # weekly, monthly, yearly
            today = timezone.now().date()
            
            if report_type == 'weekly':
                start_date = today - timedelta(days=6)
                end_date = today
            elif report_type == 'monthly':
                start_date = today.replace(day=1)
                end_date = today
            elif report_type == 'yearly':
                start_date = today.replace(month=1, day=1)
                end_date = today
            else:
                start_date = today - timedelta(days=6)
                end_date = today
            
            # Get nutrition data for the period
            nutrition_data = DailyNutritionLog.objects.filter(
                user=request.user,
                date__gte=start_date,
                date__lte=end_date
            ).values('date').annotate(
                daily_calories=Sum('calories'),
                daily_protein=Sum('protein'),
                daily_carbs=Sum('carbs'),
                daily_fats=Sum('fats')
            ).order_by('date')
            
            # Calculate totals and averages
            total_calories = sum(entry['daily_calories'] or 0 for entry in nutrition_data)
            total_protein = sum(entry['daily_protein'] or 0 for entry in nutrition_data)
            total_carbs = sum(entry['daily_carbs'] or 0 for entry in nutrition_data)
            total_fats = sum(entry['daily_fats'] or 0 for entry in nutrition_data)
            
            days_with_data = len([entry for entry in nutrition_data if entry['daily_calories']])
            avg_calories = total_calories / days_with_data if days_with_data > 0 else 0
            avg_protein = total_protein / days_with_data if days_with_data > 0 else 0
            avg_carbs = total_carbs / days_with_data if days_with_data > 0 else 0
            avg_fats = total_fats / days_with_data if days_with_data > 0 else 0
            
            # Get user targets
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                targets = {
                    'calories': user_profile.target_calories,
                    'protein': user_profile.target_protein,
                    'carbs': user_profile.target_carbs,
                    'fats': user_profile.target_fats
                }
            except UserProfile.DoesNotExist:
                targets = {'calories': 0, 'protein': 0, 'carbs': 0, 'fats': 0}
            
            # Calculate achievement rates
            achievement_rate = {
                'calories': round((avg_calories / targets['calories'] * 100) if targets['calories'] > 0 else 0, 1),
                'protein': round((avg_protein / targets['protein'] * 100) if targets['protein'] > 0 else 0, 1),
                'carbs': round((avg_carbs / targets['carbs'] * 100) if targets['carbs'] > 0 else 0, 1),
                'fats': round((avg_fats / targets['fats'] * 100) if targets['fats'] > 0 else 0, 1)
            }
            
            # Get top foods consumed
            top_foods = DailyNutritionLog.objects.filter(
                user=request.user,
                date__gte=start_date,
                date__lte=end_date
            ).values('custom_food_name').annotate(
                total_calories=Sum('calories'),
                frequency=Count('id')
            ).order_by('-total_calories')[:10]
            
            # Get meal type breakdown
            meal_breakdown = DailyNutritionLog.objects.filter(
                user=request.user,
                date__gte=start_date,
                date__lte=end_date
            ).values('meal_type').annotate(
                total_calories=Sum('calories'),
                count=Count('id')
            ).order_by('-total_calories')
            
            return Response({
                'success': True,
                'report_type': report_type,
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days_tracked': days_with_data
                },
                'summary': {
                    'total_nutrition': {
                        'calories': total_calories,
                        'protein': total_protein,
                        'carbs': total_carbs,
                        'fats': total_fats
                    },
                    'average_daily': {
                        'calories': round(avg_calories, 1),
                        'protein': round(avg_protein, 1),
                        'carbs': round(avg_carbs, 1),
                        'fats': round(avg_fats, 1)
                    },
                    'targets': targets,
                    'achievement_rate': achievement_rate
                },
                'top_foods': list(top_foods),
                'meal_breakdown': list(meal_breakdown),
                'daily_data': list(nutrition_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating nutrition report: {e}")
            return Response({
                'error': 'Failed to generate nutrition report'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
