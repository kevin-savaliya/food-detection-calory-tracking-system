from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import json

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    ACTIVITY_LEVEL_CHOICES = [
        ('sedentary', 'Sedentary (little or no exercise)'),
        ('lightly_active', 'Lightly active (light exercise/sports 1-3 days/week)'),
        ('moderately_active', 'Moderately active (moderate exercise/sports 3-5 days/week)'),
        ('very_active', 'Very active (hard exercise/sports 6-7 days a week)'),
        ('extremely_active', 'Extremely active (very hard exercise/sports & physical job)'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField(validators=[MinValueValidator(13), MaxValueValidator(120)])
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    height = models.DecimalField(max_digits=5, decimal_places=2, help_text='Height in inches')
    weight = models.DecimalField(max_digits=5, decimal_places=2, help_text='Weight in kg')
    activity_level = models.CharField(max_length=20, choices=ACTIVITY_LEVEL_CHOICES, default='sedentary')
    target_calories = models.PositiveIntegerField(default=2000)
    target_protein = models.PositiveIntegerField(default=150, help_text='Target protein in grams')
    target_carbs = models.PositiveIntegerField(default=250, help_text='Target carbs in grams')
    target_fats = models.PositiveIntegerField(default=65, help_text='Target fats in grams')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    def calculate_bmi(self):
        """Calculate BMI"""
        # Convert Decimal fields to float for arithmetic operations
        weight = float(self.weight)
        height_input = float(self.height)
        
        # Check if height input is reasonable
        # If height is less than 12 inches, it's likely just inches (e.g., 5.6 inches)
        # If height is 12-120 inches, it's likely total inches (e.g., 66 inches for 5'6")
        # If height is over 120 inches, it's likely centimeters
        
        if height_input < 12:
            # Input is just inches (e.g., 5.6 inches) - this seems too short
            # Let's assume it's feet.inches format (e.g., 5.6 means 5 feet 6 inches)
            feet = int(height_input)
            inches = (height_input - feet) * 10
            total_inches = (feet * 12) + inches
        elif height_input <= 120:
            # Input is total inches (e.g., 66 inches for 5'6")
            total_inches = height_input
        else:
            # Input is centimeters, convert to inches
            total_inches = height_input / 2.54
        
        # Convert total inches to meters
        height_cm = total_inches * 2.54
        height_m = height_cm / 100
        
        return round(weight / (height_m ** 2), 1)
    
    def calculate_bmr(self):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
        # Convert Decimal fields to float for arithmetic operations
        weight = float(self.weight)
        height_input = float(self.height)
        age = float(self.age)
        
        # Use the same height conversion logic as BMI
        if height_input < 12:
            # Input is just inches (e.g., 5.6 inches) - this seems too short
            # Let's assume it's feet.inches format (e.g., 5.6 means 5 feet 6 inches)
            feet = int(height_input)
            inches = (height_input - feet) * 10
            total_inches = (feet * 12) + inches
        elif height_input <= 120:
            # Input is total inches (e.g., 66 inches for 5'6")
            total_inches = height_input
        else:
            # Input is centimeters, convert to inches
            total_inches = height_input / 2.54
        
        # Convert total inches to centimeters for BMR calculation
        height_cm = total_inches * 2.54
        
        if self.gender == 'M':
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) - 161
        return round(bmr)
    
    def calculate_tdee(self):
        """Calculate Total Daily Energy Expenditure"""
        bmr = self.calculate_bmr()
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly_active': 1.375,
            'moderately_active': 1.55,
            'very_active': 1.725,
            'extremely_active': 1.9,
        }
        return round(bmr * activity_multipliers.get(self.activity_level, 1.2))

class FoodItem(models.Model):
    name = models.CharField(max_length=200)
    calories = models.PositiveIntegerField()
    protein = models.DecimalField(max_digits=5, decimal_places=2, help_text='Protein in grams')
    carbs = models.DecimalField(max_digits=5, decimal_places=2, help_text='Carbs in grams')
    fats = models.DecimalField(max_digits=5, decimal_places=2, help_text='Fats in grams')
    fiber = models.DecimalField(max_digits=5, decimal_places=2, default=0, help_text='Fiber in grams')
    sugar = models.DecimalField(max_digits=5, decimal_places=2, default=0, help_text='Sugar in grams')
    sodium = models.DecimalField(max_digits=6, decimal_places=2, default=0, help_text='Sodium in mg')
    serving_size = models.CharField(max_length=100, default='100g')
    category = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

class FoodDetectionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detection_logs')
    image = models.ImageField(upload_to='food_images/')
    detected_foods = models.JSONField(default=list, help_text='List of detected food items')
    confidence_scores = models.JSONField(default=list, help_text='Confidence scores for detections')
    processed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.processed_at.strftime('%Y-%m-%d %H:%M')}"

class DailyNutritionLog(models.Model):
    MEAL_CHOICES = [
        ('breakfast', 'Breakfast'),
        ('lunch', 'Lunch'),
        ('dinner', 'Dinner'),
        ('snack', 'Snack'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='nutrition_logs')
    date = models.DateField()
    meal_type = models.CharField(max_length=20, choices=MEAL_CHOICES)
    food_item = models.ForeignKey(FoodItem, on_delete=models.CASCADE, null=True, blank=True)
    custom_food_name = models.CharField(max_length=200, blank=True)
    quantity = models.DecimalField(max_digits=5, decimal_places=2, default=1.0, help_text='Quantity in servings')
    calories = models.PositiveIntegerField()
    protein = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    carbs = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    fats = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.meal_type}"
    
    class Meta:
        ordering = ['-date', '-created_at']
        unique_together = ['user', 'date', 'meal_type', 'food_item', 'custom_food_name']

class NutritionGoal(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='nutrition_goals')
    date = models.DateField()
    target_calories = models.PositiveIntegerField()
    target_protein = models.PositiveIntegerField()
    target_carbs = models.PositiveIntegerField()
    target_fats = models.PositiveIntegerField()
    actual_calories = models.PositiveIntegerField(default=0)
    actual_protein = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    actual_carbs = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    actual_fats = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.date}"
    
    class Meta:
        unique_together = ['user', 'date']
        ordering = ['-date']
