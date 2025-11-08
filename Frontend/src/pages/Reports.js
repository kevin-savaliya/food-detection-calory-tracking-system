import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Calendar, 
  Target,
  Award,
  Clock,
  PieChart
} from 'lucide-react';
import api from '../api/api';

const Reports = () => {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reportType, setReportType] = useState('weekly');

  useEffect(() => {
    fetchReportData();
  }, [reportType]);

  // Listen for food detection completion to refresh data
  useEffect(() => {
    const handleFoodDetectionComplete = () => {
      fetchReportData();
    };

    window.addEventListener('foodDetectionComplete', handleFoodDetectionComplete);

    return () => {
      window.removeEventListener('foodDetectionComplete', handleFoodDetectionComplete);
    };
  }, [reportType]);

  const fetchReportData = async () => {
    try {
      setLoading(true);
      const response = await api.get(`/nutrition-reports/?type=${reportType}`);
      setReportData(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching report data:', err);
      setError('Failed to load report data');
    } finally {
      setLoading(false);
    }
  };

  const getReportTypeLabel = (type) => {
    switch (type) {
      case 'weekly': return 'Weekly Report';
      case 'monthly': return 'Monthly Report';
      case 'yearly': return 'Yearly Report';
      default: return 'Report';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2 flex items-center">
          <BarChart3 className="h-6 w-6 mr-3" />
          Nutrition Reports
        </h1>
        <p className="text-primary-100">
          Analyze your nutrition patterns and track your progress over time.
        </p>
      </div>

      {/* Report Type Selector */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">
            {getReportTypeLabel(reportType)}
          </h2>
          <div className="flex space-x-2">
            {['weekly', 'monthly', 'yearly'].map((type) => (
              <button
                key={type}
                onClick={() => setReportType(type)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  reportType === type
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {getReportTypeLabel(type)}
              </button>
            ))}
          </div>
        </div>

        {reportData && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Period Info */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                <Calendar className="h-4 w-4 mr-2" />
                Period Overview
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Start Date:</span>
                  <span className="font-medium">{formatDate(reportData.period.start_date)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">End Date:</span>
                  <span className="font-medium">{formatDate(reportData.period.end_date)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Days Tracked:</span>
                  <span className="font-medium">{reportData.period.days_tracked}</span>
                </div>
              </div>
            </div>

            {/* Achievement Rate */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                <Target className="h-4 w-4 mr-2" />
                Goal Achievement
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <p className="text-sm text-gray-600">Calories</p>
                  <p className="text-lg font-bold text-gray-900">
                    {reportData.summary.achievement_rate.calories}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Protein</p>
                  <p className="text-lg font-bold text-gray-900">
                    {reportData.summary.achievement_rate.protein}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Summary Cards */}
      {reportData ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-red-100 rounded-lg">
                <TrendingUp className="h-6 w-6 text-red-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Calories</p>
                <p className="text-2xl font-bold text-gray-900">
                  {reportData.summary.total_nutrition.calories}
                </p>
                <p className="text-xs text-gray-500">
                  Avg: {reportData.summary.average_daily.calories}/day
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Target className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Protein</p>
                <p className="text-2xl font-bold text-gray-900">
                  {reportData.summary.total_nutrition.protein}g
                </p>
                <p className="text-xs text-gray-500">
                  Avg: {reportData.summary.average_daily.protein}g/day
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <BarChart3 className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Carbs</p>
                <p className="text-2xl font-bold text-gray-900">
                  {reportData.summary.total_nutrition.carbs}g
                </p>
                <p className="text-xs text-gray-500">
                  Avg: {reportData.summary.average_daily.carbs}g/day
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <PieChart className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Fats</p>
                <p className="text-2xl font-bold text-gray-900">
                  {reportData.summary.total_nutrition.fats}g
                </p>
                <p className="text-xs text-gray-500">
                  Avg: {reportData.summary.average_daily.fats}g/day
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12">
          <BarChart3 className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Report Data Available</h3>
          <p className="text-gray-500 mb-4">
            No nutrition data available for the selected period.
          </p>
          <p className="text-sm text-gray-400">
            Start logging your meals to see reports and analytics.
          </p>
        </div>
      )}

      {/* Top Foods */}
      {reportData && reportData.top_foods && reportData.top_foods.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Award className="h-5 w-5 mr-2" />
            Top Foods Consumed
          </h2>
          <div className="space-y-3">
            {reportData.top_foods.slice(0, 10).map((food, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center mr-3">
                    <span className="text-sm font-bold text-primary-600">#{index + 1}</span>
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{food.custom_food_name}</p>
                    <p className="text-sm text-gray-600">
                      {food.frequency} time{food.frequency !== 1 ? 's' : ''} consumed
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900">{food.total_calories} cal</p>
                  <p className="text-sm text-gray-600">total</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Meal Breakdown */}
      {reportData && reportData.meal_breakdown && reportData.meal_breakdown.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Clock className="h-5 w-5 mr-2" />
            Meal Type Breakdown
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {reportData.meal_breakdown.map((meal, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg text-center">
                <h3 className="font-medium text-gray-900 capitalize mb-2">
                  {meal.meal_type}
                </h3>
                <p className="text-2xl font-bold text-primary-600 mb-1">
                  {meal.total_calories}
                </p>
                <p className="text-sm text-gray-600">calories</p>
                <p className="text-xs text-gray-500 mt-1">
                  {meal.count} meal{meal.count !== 1 ? 's' : ''}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Daily Data Chart */}
      {reportData && reportData.daily_data && reportData.daily_data.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Daily Nutrition Trends</h2>
          <div className="space-y-4">
            {reportData.daily_data.map((day, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-gray-900">
                    {formatDate(day.date)}
                  </h3>
                  <div className="flex items-center space-x-4 text-sm text-gray-600">
                    <span>{day.daily_calories || 0} cal</span>
                    <span>•</span>
                    <span>{day.daily_protein || 0}g protein</span>
                    <span>•</span>
                    <span>{day.daily_carbs || 0}g carbs</span>
                    <span>•</span>
                    <span>{day.daily_fats || 0}g fats</span>
                  </div>
                </div>
                
                {/* Progress bars for each nutrient */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Calories</span>
                    <span>{day.daily_calories || 0}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full"
                      style={{ 
                        width: `${Math.min(((day.daily_calories || 0) / (reportData.summary.targets.calories || 1)) * 100, 100)}%` 
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;