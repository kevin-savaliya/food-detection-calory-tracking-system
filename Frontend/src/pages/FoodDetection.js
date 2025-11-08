import React, { useState, useRef } from 'react';
import { Camera, Upload, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react';
import api from '../api/api';

const FoodDetection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }

      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await api.post('/detect-food/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
      console.log('Detection Result:', response.data);
      
      // Trigger dashboard refresh by dispatching a custom event
      window.dispatchEvent(new CustomEvent('foodDetectionComplete', {
        detail: response.data
      }));
    } catch (err) {
      console.error('Error uploading image for detection:', err);
      setError(err.response?.data?.error || 'Failed to detect food. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetDetection = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'text-green-600 bg-green-100';
    if (confidence >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-primary-600 to-secondary-600 rounded-lg p-6 text-white">
        <h1 className="text-2xl font-bold mb-2 flex items-center">
          <Camera className="h-6 w-6 mr-3" />
          Food Detection
        </h1>
        <p className="text-primary-100">
          Upload an image of your food and let AI identify it for you.
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-center">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8">
            {preview ? (
              <div className="space-y-4">
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="mx-auto max-h-64 rounded-lg shadow-md"
                />
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="btn-primary flex items-center"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Detecting...
                      </>
                    ) : (
                      <>
                        <Upload className="h-4 w-4 mr-2" />
                        Detect Food
                      </>
                    )}
                  </button>
                  <button
                    onClick={resetDetection}
                    className="btn-secondary flex items-center"
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto h-16 w-16 bg-gray-100 rounded-full flex items-center justify-center">
                  <Camera className="h-8 w-8 text-gray-400" />
                </div>
                <div>
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="btn-primary inline-flex items-center">
                      <Upload className="h-4 w-4 mr-2" />
                      Choose Image
                    </span>
                    <input
                      ref={fileInputRef}
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                </div>
                <p className="text-sm text-gray-500">
                  Upload a clear image of your food (JPG, PNG, GIF up to 10MB)
                </p>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 bg-danger-50 border border-danger-200 text-danger-700 px-4 py-3 rounded-lg flex items-center">
            <AlertCircle className="h-5 w-5 mr-2" />
            {error}
          </div>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Detection Results */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
              Detection Results
            </h2>
            
            {result.detected_foods && result.detected_foods.length > 0 ? (
              <div className="space-y-4">
                {result.detected_foods.map((foodName, index) => {
                  const detailedFood = result.detailed_analysis && result.detailed_analysis[index] 
                    ? result.detailed_analysis[index] 
                    : { name: foodName, confidence: result.confidence_scores[index] || 0 };
                  
                  return (
                    <div key={index} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-medium text-gray-900">{detailedFood.name}</h3>
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(detailedFood.confidence)}`}>
                          {detailedFood.confidence}% confident
                        </span>
                      </div>
                      
                      {detailedFood.nutrition && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                          <div className="text-center p-3 bg-red-50 rounded-lg">
                            <p className="text-sm text-red-600 font-medium">Calories</p>
                            <p className="text-lg font-bold text-red-700">{detailedFood.nutrition.calories || 0}</p>
                          </div>
                          <div className="text-center p-3 bg-blue-50 rounded-lg">
                            <p className="text-sm text-blue-600 font-medium">Protein</p>
                            <p className="text-lg font-bold text-blue-700">{detailedFood.nutrition.protein || 0}g</p>
                          </div>
                          <div className="text-center p-3 bg-yellow-50 rounded-lg">
                            <p className="text-sm text-yellow-600 font-medium">Carbs</p>
                            <p className="text-lg font-bold text-yellow-700">{detailedFood.nutrition.carbs || 0}g</p>
                          </div>
                          <div className="text-center p-3 bg-green-50 rounded-lg">
                            <p className="text-sm text-green-600 font-medium">Fats</p>
                            <p className="text-lg font-bold text-green-700">{detailedFood.nutrition.fats || 0}g</p>
                          </div>
                        </div>
                      )}
                      
                      {detailedFood.description && (
                        <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                          <p className="text-sm text-gray-700">{detailedFood.description}</p>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No food items detected in this image.</p>
                <p className="text-sm text-gray-400 mt-1">
                  Try uploading a clearer image with visible food items.
                </p>
              </div>
            )}
          </div>

          {/* Nutrition Summary */}
          {result.saved_foods && result.saved_foods.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Total Nutrition (Saved to Log)</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-red-50 rounded-lg">
                  <p className="text-sm text-red-600 font-medium">Total Calories</p>
                  <p className="text-2xl font-bold text-red-700">
                    {result.saved_foods.reduce((sum, food) => sum + (food.calories || 0), 0)}
                  </p>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-600 font-medium">Total Protein</p>
                  <p className="text-2xl font-bold text-blue-700">
                    {result.saved_foods.reduce((sum, food) => sum + (food.protein || 0), 0)}g
                  </p>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <p className="text-sm text-yellow-600 font-medium">Total Carbs</p>
                  <p className="text-2xl font-bold text-yellow-700">
                    {result.saved_foods.reduce((sum, food) => sum + (food.carbs || 0), 0)}g
                  </p>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-600 font-medium">Total Fats</p>
                  <p className="text-2xl font-bold text-green-700">
                    {result.saved_foods.reduce((sum, food) => sum + (food.fats || 0), 0)}g
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Image Description */}
          {result.image_description && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Image Analysis</h2>
              <p className="text-gray-700">{result.image_description}</p>
            </div>
          )}

          {/* Success Message */}
          {result.saved_foods && result.saved_foods.length > 0 && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 mr-2" />
                <div>
                  <p className="font-medium">Successfully saved to your nutrition log!</p>
                  <p className="text-sm">
                    {result.saved_foods.length} food item(s) added to today's {result.saved_foods[0]?.meal_type || 'log'}.
                  </p>
                  <div className="mt-2">
                    <p className="text-xs">Saved items:</p>
                    <ul className="text-xs mt-1">
                      {result.saved_foods.map((food, index) => (
                        <li key={index}>
                          • {food.food_name} ({food.calories} cal, {food.protein}g protein, {food.carbs}g carbs, {food.fats}g fats)
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Overall Detection Info */}
          {result.overall_confidence && (
            <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 mr-2" />
                <div>
                  <p className="font-medium">Overall Detection Confidence</p>
                  <p className="text-sm">
                    {Math.round(result.overall_confidence)}% confident in this analysis
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Detection Method */}
          {result.detection_log_id && (
            <div className="bg-gray-50 border border-gray-200 text-gray-700 px-4 py-3 rounded-lg">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 mr-2" />
                <div>
                  <p className="font-medium">Detection Logged</p>
                  <p className="text-sm">
                    Detection ID: {result.detection_log_id}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FoodDetection;
