import React, { useState, useEffect } from 'react';
import { 
  Calendar, 
  Plus, 
  Trash2, 
  Edit, 
  Clock,
  Target,
  TrendingUp,
  BookOpen
} from 'lucide-react';
import api from '../api/api';

const NutritionLog = () => {
  const [nutritionData, setNutritionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [showAddForm, setShowAddForm] = useState(false);
  const [deletingId, setDeletingId] = useState(null);

  const [newEntry, setNewEntry] = useState({
    food_name: '',
    meal_type: 'snack',
    quantity: 1,
    calories: '',
    protein: '',
    carbs: '',
    fats: '',
    notes: ''
  });

  useEffect(() => {
    fetchNutritionData();
  }, [selectedDate]);

  // Listen for food detection completion to refresh data
  useEffect(() => {
    const handleFoodDetectionComplete = () => {
      fetchNutritionData();
    };

    window.addEventListener('foodDetectionComplete', handleFoodDetectionComplete);

    return () => {
      window.removeEventListener('foodDetectionComplete', handleFoodDetectionComplete);
    };
  }, []);

  const fetchNutritionData = async () => {
    try {
      setLoading(true);
      const dateStr = selectedDate.toISOString().split('T')[0];
      const response = await api.get(`/nutrition-log-enhanced/?date=${dateStr}`);
      setNutritionData(response.data);
      setError('');
    } catch (err) {
      console.error('Error fetching nutrition log:', err);
      setError('Failed to load nutrition log');
    } finally {
      setLoading(false);
    }
  };

  const handleAddEntry = async (e) => {
    e.preventDefault();
    try {
      const entryData = {
        ...newEntry,
        date: selectedDate.toISOString().split('T')[0],
        calories: parseFloat(newEntry.calories) || 0,
        protein: parseFloat(newEntry.protein) || 0,
        carbs: parseFloat(newEntry.carbs) || 0,
        fats: parseFloat(newEntry.fats) || 0
      };

      await api.post('/nutrition-log-enhanced/', entryData);
      setShowAddForm(false);
      setNewEntry({
        food_name: '',
        meal_type: 'snack',
        quantity: 1,
        calories: '',
        protein: '',
        carbs: '',
        fats: '',
        notes: ''
      });
      fetchNutritionData();
    } catch (err) {
      console.error('Error adding nutrition entry:', err);
      setError('Failed to add nutrition entry');
    }
  };

  const handleDeleteEntry = async (entryId) => {
    if (!window.confirm('Are you sure you want to delete this entry?')) {
      return;
    }

    try {
      setDeletingId(entryId);
      await api.delete('/nutrition-log-enhanced/', {
        data: { entry_id: entryId }
      });
      fetchNutritionData();
    } catch (err) {
      console.error('Error deleting nutrition entry:', err);
      setError('Failed to delete nutrition entry');
    } finally {
      setDeletingId(null);
    }
  };

  const mealTypes = [
    { value: 'breakfast', label: '🌅 Breakfast', color: 'bg-orange-100 text-orange-800' },
    { value: 'lunch', label: '🌞 Lunch', color: 'bg-blue-100 text-blue-800' },
    { value: 'dinner', label: '🌙 Dinner', color: 'bg-purple-100 text-purple-800' },
    { value: 'snack', label: '🍿 Snack', color: 'bg-green-100 text-green-800' }
  ];

  const getMealTypeConfig = (mealType) => {
    return mealTypes.find(m => m.value === mealType) || mealTypes[3];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2 flex items-center">
          <BookOpen className="h-6 w-6 mr-3" />
          Nutrition Log
        </h1>
        <p className="text-primary-100">
          Track your daily nutrition intake and monitor your progress.
        </p>
      </div>

      {/* Date Selector and Add Button */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <Calendar className="h-5 w-5 mr-2 text-gray-600" />
              <label htmlFor="date-select" className="text-sm font-medium text-gray-700">
                Select Date:
              </label>
            </div>
            <input
              id="date-select"
              type="date"
              value={selectedDate.toISOString().split('T')[0]}
              onChange={(e) => setSelectedDate(new Date(e.target.value))}
              className="input-field w-auto"
            />
          </div>
          <button
            onClick={() => setShowAddForm(true)}
            className="btn-primary flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Entry
          </button>
        </div>

        {error && (
          <div className="bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {/* Daily Summary */}
        {nutritionData ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                <Target className="h-4 w-4 mr-2" />
                Today's Totals
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-red-50 rounded-lg">
                  <p className="text-sm text-red-600 font-medium">Calories</p>
                  <p className="text-xl font-bold text-red-700">
                    {nutritionData.total_nutrition.calories}
                  </p>
                </div>
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-600 font-medium">Protein</p>
                  <p className="text-xl font-bold text-blue-700">
                    {nutritionData.total_nutrition.protein}g
                  </p>
                </div>
                <div className="text-center p-3 bg-yellow-50 rounded-lg">
                  <p className="text-sm text-yellow-600 font-medium">Carbs</p>
                  <p className="text-xl font-bold text-yellow-700">
                    {nutritionData.total_nutrition.carbs}g
                  </p>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-600 font-medium">Fats</p>
                  <p className="text-xl font-bold text-green-700">
                    {nutritionData.total_nutrition.fats}g
                  </p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2" />
                Progress
              </h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Calories</span>
                    <span>{nutritionData.progress.calories}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full"
                      style={{ width: `${Math.min(nutritionData.progress.calories, 100)}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Protein</span>
                    <span>{nutritionData.progress.protein}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${Math.min(nutritionData.progress.protein, 100)}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Carbs</span>
                    <span>{nutritionData.progress.carbs}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-500 h-2 rounded-full"
                      style={{ width: `${Math.min(nutritionData.progress.carbs, 100)}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Fats</span>
                    <span>{nutritionData.progress.fats}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${Math.min(nutritionData.progress.fats, 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No nutrition data available for this date.</p>
            <p className="text-sm text-gray-400 mt-1">
              Start by detecting some food or adding a meal manually.
            </p>
          </div>
        )}
      </div>

      {/* Meals by Type */}
      {nutritionData ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {mealTypes.map((mealType) => {
            const meals = nutritionData.meals[mealType.value] || [];
            const config = getMealTypeConfig(mealType.value);
            
            return (
              <div key={mealType.value} className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <span className={`px-2 py-1 rounded-full text-sm font-medium ${config.color} mr-3`}>
                    {config.label}
                  </span>
                  ({meals.length} items)
                </h3>

                {meals.length > 0 ? (
                  <div className="space-y-3">
                    {meals.map((meal) => (
                      <div key={meal.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="font-medium text-gray-900">{meal.food_name}</h4>
                            <span className="text-sm font-semibold text-gray-700">
                              {meal.calories} cal
                            </span>
                          </div>
                          <div className="flex items-center space-x-4 text-xs text-gray-600">
                            <span>{meal.protein}g protein</span>
                            <span>•</span>
                            <span>{meal.carbs}g carbs</span>
                            <span>•</span>
                            <span>{meal.fats}g fats</span>
                            <span>•</span>
                            <span>Qty: {meal.quantity}</span>
                          </div>
                          {meal.notes && (
                            <p className="text-xs text-gray-500 mt-1 italic">{meal.notes}</p>
                          )}
                          <div className="flex items-center text-xs text-gray-500 mt-1">
                            <Clock className="h-3 w-3 mr-1" />
                            {new Date(meal.created_at).toLocaleTimeString()}
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteEntry(meal.id)}
                          disabled={deletingId === meal.id}
                          className="ml-4 p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                        >
                          {deletingId === meal.id ? (
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600"></div>
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No {mealType.value} logged yet.</p>
                    <p className="text-sm">Add an entry to start tracking!</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <BookOpen className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Nutrition Data</h3>
          <p className="text-gray-500 mb-4">
            No nutrition data available for {selectedDate.toLocaleDateString()}.
          </p>
          <button
            onClick={() => setShowAddForm(true)}
            className="btn-primary"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Your First Entry
          </button>
        </div>
      )}

      {/* Add Entry Modal */}
      {showAddForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Add Nutrition Entry</h2>
              
              <form onSubmit={handleAddEntry} className="space-y-4">
                <div>
                  <label htmlFor="food_name" className="form-label">Food Name</label>
                  <input
                    id="food_name"
                    name="food_name"
                    type="text"
                    value={newEntry.food_name}
                    onChange={(e) => setNewEntry({ ...newEntry, food_name: e.target.value })}
                    className="input-field"
                    required
                    placeholder="e.g., Apple, Chicken Breast"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="meal_type" className="form-label">Meal Type</label>
                    <select
                      id="meal_type"
                      name="meal_type"
                      value={newEntry.meal_type}
                      onChange={(e) => setNewEntry({ ...newEntry, meal_type: e.target.value })}
                      className="input-field"
                    >
                      {mealTypes.map((type) => (
                        <option key={type.value} value={type.value}>
                          {type.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label htmlFor="quantity" className="form-label">Quantity</label>
                    <input
                      id="quantity"
                      name="quantity"
                      type="number"
                      min="0.1"
                      step="0.1"
                      value={newEntry.quantity}
                      onChange={(e) => setNewEntry({ ...newEntry, quantity: parseFloat(e.target.value) || 1 })}
                      className="input-field"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="calories" className="form-label">Calories</label>
                    <input
                      id="calories"
                      name="calories"
                      type="number"
                      min="0"
                      step="0.1"
                      value={newEntry.calories}
                      onChange={(e) => setNewEntry({ ...newEntry, calories: e.target.value })}
                      className="input-field"
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="protein" className="form-label">Protein (g)</label>
                    <input
                      id="protein"
                      name="protein"
                      type="number"
                      min="0"
                      step="0.1"
                      value={newEntry.protein}
                      onChange={(e) => setNewEntry({ ...newEntry, protein: e.target.value })}
                      className="input-field"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="carbs" className="form-label">Carbs (g)</label>
                    <input
                      id="carbs"
                      name="carbs"
                      type="number"
                      min="0"
                      step="0.1"
                      value={newEntry.carbs}
                      onChange={(e) => setNewEntry({ ...newEntry, carbs: e.target.value })}
                      className="input-field"
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="fats" className="form-label">Fats (g)</label>
                    <input
                      id="fats"
                      name="fats"
                      type="number"
                      min="0"
                      step="0.1"
                      value={newEntry.fats}
                      onChange={(e) => setNewEntry({ ...newEntry, fats: e.target.value })}
                      className="input-field"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="notes" className="form-label">Notes (optional)</label>
                  <textarea
                    id="notes"
                    name="notes"
                    value={newEntry.notes}
                    onChange={(e) => setNewEntry({ ...newEntry, notes: e.target.value })}
                    className="input-field"
                    rows="3"
                    placeholder="Any additional notes..."
                  />
                </div>

                <div className="flex justify-end space-x-4 pt-4">
                  <button
                    type="button"
                    onClick={() => setShowAddForm(false)}
                    className="btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn-primary"
                  >
                    Add Entry
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NutritionLog;
