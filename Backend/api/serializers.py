from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, FoodItem, FoodDetectionLog, DailyNutritionLog, NutritionGoal

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']

class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField(
        error_messages={
            'blank': 'Username is required.',
        }
    )
    password = serializers.CharField(
        error_messages={
            'blank': 'Password is required.',
        }
    )

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    bmi = serializers.SerializerMethodField()
    bmr = serializers.SerializerMethodField()
    tdee = serializers.SerializerMethodField()
    
    class Meta:
        model = UserProfile
        fields = [
            'id', 'user', 'age', 'gender', 'height', 'weight', 'activity_level',
            'target_calories', 'target_protein', 'target_carbs', 'target_fats',
            'bmi', 'bmr', 'tdee', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_bmi(self, obj):
        return obj.calculate_bmi()
    
    def get_bmr(self, obj):
        return obj.calculate_bmr()
    
    def get_tdee(self, obj):
        return obj.calculate_tdee()

class FoodItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = FoodItem
        fields = [
            'id', 'name', 'calories', 'protein', 'carbs', 'fats',
            'fiber', 'sugar', 'sodium', 'serving_size', 'category', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class FoodDetectionLogSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = FoodDetectionLog
        fields = [
            'id', 'user', 'image', 'detected_foods', 'confidence_scores', 'processed_at'
        ]
        read_only_fields = ['id', 'processed_at']

class DailyNutritionLogSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    food_item = FoodItemSerializer(read_only=True)
    food_item_id = serializers.IntegerField(write_only=True, required=False, allow_null=True)
    
    class Meta:
        model = DailyNutritionLog
        fields = [
            'id', 'user', 'date', 'meal_type', 'food_item', 'food_item_id',
            'custom_food_name', 'quantity', 'calories', 'protein', 'carbs', 'fats',
            'notes', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def create(self, validated_data):
        food_item_id = validated_data.pop('food_item_id', None)
        if food_item_id:
            try:
                validated_data['food_item'] = FoodItem.objects.get(id=food_item_id)
            except FoodItem.DoesNotExist:
                raise serializers.ValidationError("Food item not found")
        return super().create(validated_data)

class NutritionGoalSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    calories_progress = serializers.SerializerMethodField()
    protein_progress = serializers.SerializerMethodField()
    carbs_progress = serializers.SerializerMethodField()
    fats_progress = serializers.SerializerMethodField()
    
    class Meta:
        model = NutritionGoal
        fields = [
            'id', 'user', 'date', 'target_calories', 'target_protein', 'target_carbs', 'target_fats',
            'actual_calories', 'actual_protein', 'actual_carbs', 'actual_fats',
            'calories_progress', 'protein_progress', 'carbs_progress', 'fats_progress',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_calories_progress(self, obj):
        if obj.target_calories > 0:
            return round((obj.actual_calories / obj.target_calories) * 100, 1)
        return 0
    
    def get_protein_progress(self, obj):
        if obj.target_protein > 0:
            return round((float(obj.actual_protein) / obj.target_protein) * 100, 1)
        return 0
    
    def get_carbs_progress(self, obj):
        if obj.target_carbs > 0:
            return round((float(obj.actual_carbs) / obj.target_carbs) * 100, 1)
        return 0
    
    def get_fats_progress(self, obj):
        if obj.target_fats > 0:
            return round((float(obj.actual_fats) / obj.target_fats) * 100, 1)
        return 0

class FoodDetectionRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    
    def validate_image(self, value):
        # Validate image size (max 10MB)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image file too large ( > 10MB )")
        return value

class NutritionSummarySerializer(serializers.Serializer):
    date = serializers.DateField()
    total_calories = serializers.IntegerField()
    total_protein = serializers.DecimalField(max_digits=5, decimal_places=2)
    total_carbs = serializers.DecimalField(max_digits=5, decimal_places=2)
    total_fats = serializers.DecimalField(max_digits=5, decimal_places=2)
    target_calories = serializers.IntegerField()
    target_protein = serializers.IntegerField()
    target_carbs = serializers.IntegerField()
    target_fats = serializers.IntegerField()
    calories_progress = serializers.FloatField()
    protein_progress = serializers.FloatField()
    carbs_progress = serializers.FloatField()
    fats_progress = serializers.FloatField()
    meal_breakdown = serializers.DictField()

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True, 
        min_length=8,
        error_messages={
            'min_length': 'Password must be at least 8 characters long.',
            'blank': 'Password is required.',
        }
    )
    password_confirm = serializers.CharField(
        write_only=True,
        error_messages={
            'blank': 'Please confirm your password.',
        }
    )
    
    # Profile fields
    height = serializers.DecimalField(
        max_digits=5, 
        decimal_places=2,
        min_value=0,
        error_messages={
            'required': 'Height is required.',
            'min_value': 'Height must be a positive number.',
            'invalid': 'Please enter a valid height in inches.',
        }
    )
    weight = serializers.DecimalField(
        max_digits=5, 
        decimal_places=2,
        min_value=0,
        error_messages={
            'required': 'Weight is required.',
            'min_value': 'Weight must be a positive number.',
            'invalid': 'Please enter a valid weight in kg.',
        }
    )
    age = serializers.IntegerField(
        min_value=1,
        max_value=120,
        error_messages={
            'required': 'Age is required.',
            'min_value': 'Age must be at least 1 year.',
            'max_value': 'Age cannot exceed 120 years.',
            'invalid': 'Please enter a valid age.',
        }
    )
    gender = serializers.ChoiceField(
        choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Other')],
        error_messages={
            'required': 'Gender is required.',
            'invalid_choice': 'Please select a valid gender.',
        }
    )
    activity_level = serializers.ChoiceField(
        choices=[
            ('sedentary', 'Sedentary (little or no exercise)'),
            ('lightly_active', 'Lightly Active (light exercise 1-3 days/week)'),
            ('moderately_active', 'Moderately Active (moderate exercise 3-5 days/week)'),
            ('very_active', 'Very Active (hard exercise 6-7 days/week)'),
            ('extremely_active', 'Extremely Active (very hard exercise, physical job)')
        ],
        error_messages={
            'required': 'Activity level is required.',
            'invalid_choice': 'Please select a valid activity level.',
        }
    )
    
    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'password_confirm', 
            'first_name', 'last_name', 'height', 'weight', 'age', 
            'gender', 'activity_level'
        ]
        extra_kwargs = {
            'username': {
                'error_messages': {
                    'unique': 'This username is already taken. Please choose a different one.',
                    'blank': 'Username is required.',
                }
            },
            'email': {
                'error_messages': {
                    'unique': 'This email is already registered. Please use a different email or try logging in.',
                    'blank': 'Email is required.',
                    'invalid': 'Please enter a valid email address.',
                }
            },
            'first_name': {
                'error_messages': {
                    'blank': 'First name is required.',
                }
            },
            'last_name': {
                'error_messages': {
                    'blank': 'Last name is required.',
                }
            },
        }
    
    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': 'Passwords do not match. Please make sure both passwords are identical.'
            })
        
        return data
    
    def create(self, validated_data):
        # Extract profile data
        profile_data = {
            'height': validated_data.pop('height'),
            'weight': validated_data.pop('weight'),
            'age': validated_data.pop('age'),
            'gender': validated_data.pop('gender'),
            'activity_level': validated_data.pop('activity_level'),
        }
        
        # Remove password_confirm
        validated_data.pop('password_confirm')
        
        # Create user
        user = User.objects.create_user(**validated_data)
        
        # Create user profile
        UserProfile.objects.create(user=user, **profile_data)
        
        return user 