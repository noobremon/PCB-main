import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

function App() {
  // Manual inspection state
  const [defectiveImage, setDefectiveImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [inspectionResults, setInspectionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // System state
  const [stats, setStats] = useState(null);
  const [sessionStats, setSessionStats] = useState({
    total_inspected: 0,
    defective_count: 0,
    pass_rate: 0,
    avg_processing_time: 0
  });
  const [history, setHistory] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const [modelStatus, setModelStatus] = useState({ trained: false, training: false });
  
  // Real-time inspection state
  const [realtimeMode, setRealtimeMode] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [cameraConnected, setCameraConnected] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [workflowState, setWorkflowState] = useState({
    state: 'idle',
    message: 'Ready to start inspection',
    progress: 0
  });
  const [realtimeResults, setRealtimeResults] = useState([]);
  const [cameraPreview, setCameraPreview] = useState(null);
  const [cameraSettings, setCameraSettings] = useState({
    resolution: '1280x720',
    fps: 30,
    quality: 70
  });
  
  // WebSocket connection
  const ws = useRef(null);
  const previewInterval = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (previewInterval.current) {
        clearInterval(previewInterval.current);
        previewInterval.current = null;
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    fetchStats();
    fetchHistory();
    fetchCameras();
    
    // No automatic refresh - only fetch once on component mount
    
    // Setup WebSocket for real-time updates (only in realtime mode)
    if (realtimeMode) {
      setupWebSocket();
    }
    
    // Cleanup function
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (previewInterval.current) {
        clearInterval(previewInterval.current);
        previewInterval.current = null;
      }
    };
  }, [realtimeMode]);

  // Start camera preview when camera is connected in real-time mode
  useEffect(() => {
    if (realtimeMode && cameraConnected && !previewInterval.current) {
      startCameraPreview();
    } else if ((!realtimeMode || !cameraConnected) && previewInterval.current) {
      stopCameraPreview();
    }
  }, [realtimeMode, cameraConnected]);

  // Setup WebSocket connection
  const setupWebSocket = () => {
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/api/realtime/ws';
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (realtimeMode) {
          setupWebSocket();
        }
      }, 3000);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  // Handle WebSocket messages
  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'state_change':
        setWorkflowState(prev => ({ ...prev, ...message.data }));
        break;
      case 'inspection_result':
        setRealtimeResults(prev => [message.data, ...prev.slice(0, 19)]); // Keep last 20
        fetchWorkflowStats(); // Update stats
        break;
      case 'error':
        setError(message.data.message);
        break;
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/pcb/stats`);
      setStats(response.data);
      setModelStatus(prev => ({
        ...prev,
        trained: response.data.reference_model_available
      }));
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchHistory = async () => {
    console.log('=== FETCHING HISTORY START ===');
    
    try {
      // Get real data from backend
      console.log('🔄 Fetching real inspection data from backend...');
      const response = await axios.get(`${API_BASE_URL}/api/pcb/inspections`);
      
      if (response.data && Array.isArray(response.data) && response.data.length > 0) {
        console.log('✅ Got real data from backend:', response.data.length, 'inspections');
        
        // Sort by timestamp (newest first)
        const sortedData = response.data.sort((a, b) => 
          new Date(b.timestamp) - new Date(a.timestamp)
        );
        
        console.log('📋 First inspection:', {
          filename: sortedData[0].filename,
          timestamp: sortedData[0].timestamp,
          is_defective: sortedData[0].is_defective
        });
        
        setHistory(sortedData);
        console.log('=== FETCHING HISTORY COMPLETE (REAL DATA) ===');
      } else {
        console.log('📭 No real data available, showing empty state');
        setHistory([]);
      }
    } catch (error) {
      console.log('❌ Backend fetch failed, showing empty state:', error.message);
      setHistory([]);
    }
  };

  // Fetch available cameras
  const fetchCameras = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/pcb/realtime/available-cameras`);
      setCameras(response.data.cameras || []);
      if (response.data.cameras && response.data.cameras.length > 0) {
        setSelectedCamera(response.data.cameras[0].id);
      }
    } catch (error) {
      console.error('Error fetching cameras:', error);
      setError('Failed to fetch available cameras. Make sure the backend server is running.');
    }
  };

  const fetchWorkflowStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/realtime/workflow/stats`);
      setSessionStats(response.data.session_stats);
      setRealtimeResults(response.data.recent_results.slice(0, 20));
    } catch (error) {
      console.error('Error fetching workflow stats:', error);
    }
  };

  const trainModel = async () => {
    setModelStatus(prev => ({ ...prev, training: true }));
    setError(null);
    
    try {
      await axios.post(`${API_BASE_URL}/api/pcb/train`);
      setModelStatus({ trained: true, training: false });
      fetchStats();
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to train model');
      setModelStatus(prev => ({ ...prev, training: false }));
    }
  };

  const connectCamera = async () => {
    try {
      setError(null);
      
      // Show loading state
      const response = await axios.post(`${API_BASE_URL}/api/realtime/camera/connect`, {}, {
        params: { camera_type: selectedCamera },
        timeout: 10000 // 10 second timeout
      });
      
      // Check if the response indicates an error
      if (response.data && response.data.status === 'error') {
        // Handle Baumer camera specific errors
        if (response.data.requires_baumer) {
          setError(
            <div>
              <p style={{ marginBottom: '10px' }}>🔌 Baumer Camera Not Found</p>
              <p style={{ marginBottom: '10px' }}>{response.data.message}</p>
              {response.data.suggestion && (
                <p style={{ marginBottom: '10px', fontWeight: 'bold' }}>{response.data.suggestion}</p>
              )}
              {response.data.details && (
                <div style={{ 
                  marginTop: '10px', 
                  padding: '10px', 
                  backgroundColor: '#f8f9fa', 
                  borderRadius: '4px',
                  whiteSpace: 'pre-line'
                }}>
                  {response.data.details}
                </div>
              )}
            </div>
          );
        } else {
          // Handle other camera errors
          setError(`Camera error: ${response.data.message}`);
        }
        setCameraConnected(false);
        stopCameraPreview();
        return;
      }
      
      // If we get here, connection was successful
      setCameraConnected(true);
      
      // Start optimized camera preview
      startCameraPreview();
      
      // Log performance info
      console.log('🎥 Camera connected with optimized settings:');
      console.log('- Preview FPS: ~10');
      console.log('- Quality: 70%');
      console.log('- Resolution: 1280x720');
      console.log('- Latency: ~100ms');
      
    } catch (error) {
      console.error('Camera connection error:', error);
      
      // Extract error message from response or use default
      const errorData = error.response?.data || {};
      let errorMessage = errorData.message || error.message || 'Failed to connect to camera';
      
      // Special handling for Baumer camera not connected
      if (selectedCamera === 'baumer' || errorMessage.includes('Baumer')) {
        setError(
          <div>
            <p style={{ marginBottom: '10px' }}>🔌 Baumer Camera Connection Required</p>
            <p style={{ marginBottom: '10px' }}>
              Unable to connect to the Baumer camera. Please ensure:
            </p>
            <ul style={{ marginLeft: '20px', marginBottom: '10px' }}>
              <li>The Baumer camera is properly connected to your computer</li>
              <li>The Baumer GenICam drivers are installed</li>
              <li>No other application is using the camera</li>
              <li>The camera is powered on</li>
            </ul>
            <p style={{ fontStyle: 'italic' }}>
              Error details: {errorMessage}
            </p>
          </div>
        );
      } else {
        // For other camera types or general errors
        setError(`Camera connection error: ${errorMessage}`);
      }
      
      setCameraConnected(false);
      stopCameraPreview();
    }
  };

  const disconnectCamera = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/realtime/camera/disconnect`);
      setCameraConnected(false);
      stopCameraPreview();
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to disconnect camera');
    }
  };

  const refreshCamera = async () => {
    try {
      // Stop current preview
      stopCameraPreview();
      
      // Disconnect and reconnect to force refresh
      await axios.post(`${API_BASE_URL}/api/realtime/camera/disconnect`);
      await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
      
      const response = await axios.post(`${API_BASE_URL}/api/realtime/camera/connect`, {}, {
        params: { camera_type: selectedCamera }
      });
      
      setCameraConnected(true);
      setError(null);
      
      // Wait a bit more before starting preview
      await new Promise(resolve => setTimeout(resolve, 500));
      startCameraPreview();
      
      console.log('🔄 Camera refreshed successfully - should show live feed now');
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to refresh camera');
    }
  };

  const startCameraPreview = () => {
    if (previewInterval.current || !cameraConnected) return;
    
    previewInterval.current = setInterval(async () => {
      try {
        // Add cache-busting timestamp to ensure fresh frames
        const timestamp = Date.now();
        const response = await axios.get(`${API_BASE_URL}/api/realtime/camera/preview?t=${timestamp}`, { 
          responseType: 'blob',
          headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
          }
        });
        const imageUrl = URL.createObjectURL(response.data);
        
        // Clean up previous image URL to prevent memory leaks
        if (cameraPreview) {
          URL.revokeObjectURL(cameraPreview);
        }
        
        setCameraPreview(imageUrl);
      } catch (error) {
        console.error('Error getting camera preview:', error);
        // Set a placeholder image on error
        setCameraPreview(null);
      }
    }, 500); // Update every 500ms to reduce backend load
  };

  const stopCameraPreview = () => {
    if (previewInterval.current) {
      clearInterval(previewInterval.current);
      previewInterval.current = null;
    }
    
    // Clean up image URL to prevent memory leaks
    if (cameraPreview) {
      URL.revokeObjectURL(cameraPreview);
      setCameraPreview(null);
    }
  };

  const startRealtimeWorkflow = async (autoMode = false) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/realtime/workflow/start`, {}, {
        params: { auto_mode: autoMode }
      });
      setWorkflowState(prevState => ({
        ...prevState,
        ...response.data.state,
        is_running: true,
        auto_mode: autoMode
      }));
      fetchWorkflowStats();
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to start workflow');
    }
  };

  const stopRealtimeWorkflow = async () => {
    try {
      // Stop the workflow but keep the camera connected
      const response = await axios.post(`${API_BASE_URL}/api/realtime/workflow/stop`);
      
      // Update workflow state but keep camera connected
      setWorkflowState(prevState => ({
        ...prevState,
        ...response.data.state,
        is_running: false,
        state: 'idle',
        camera_connected: prevState.camera_connected // Preserve camera connection state
      }));
      
      // Ensure camera preview keeps running
      if (!previewInterval.current && cameraConnected) {
        startCameraPreview();
      }
      
      console.log('Workflow stopped, camera preview remains active');
    } catch (error) {
      console.error('Error stopping workflow:', error);
      setError(error.response?.data?.detail || 'Failed to stop workflow');
    }
  };

  const triggerInspection = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/realtime/workflow/trigger`);
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to trigger inspection');
    }
  };

  // Manual inspection functions (existing)
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setInspectionResults(null);
      setError(null);
    }
  }, []);

  const [datasetType, setDatasetType] = useState('raw');
  const [imageName, setImageName] = useState('');

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setError(null);
  };

  const handleDatasetTypeChange = (e) => {
    setDatasetType(e.target.value);
    // Clear image name when switching dataset type
    setImageName('');
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setInspectionResults(null);
      setError(null);
    }
  };

  const inspectPCB = async () => {
    if (!selectedFile) return;

    if (!modelStatus.trained) {
      setError('Please train the model first by adding good PCB images to the dataset/good/ folder and clicking "Train Model"');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/pcb/inspect`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setInspectionResults(response.data);

      // Handle visualization
      if (response.data.is_defective) {
        if (response.data.visualization_path) {
          const imgPath = response.data.visualization_path.startsWith('http')
            ? response.data.visualization_path
            : `${API_BASE_URL}${response.data.visualization_path}`;
          setDefectiveImage(`${imgPath}?t=${Date.now()}`);
        } else if (response.data.saved_filename) {
          // Fallback for backward compatibility
          const imgPath = `${API_BASE_URL}/api/pcb/defective/${response.data.saved_filename}`;
          setDefectiveImage(`${imgPath}?t=${Date.now()}`);
        } else {
          setDefectiveImage(null);
        }
      } else {
        setDefectiveImage(null);
      }

      fetchStats();
      
      // Immediately update history with the new inspection
      console.log('🔄 Adding new inspection to history:', response.data);
      const newInspection = {
        id: response.data.id,
        filename: response.data.image_name || file.name,
        timestamp: response.data.timestamp,
        is_defective: response.data.is_defective,
        confidence_score: response.data.confidence_score,
        defect_count: response.data.defects?.length || 0,
        visualization_path: response.data.visualization_path
      };
      
      setHistory(prevHistory => {
        const updatedHistory = [newInspection, ...prevHistory];
        console.log('✅ Updated history with new inspection:', updatedHistory.length, 'total');
        return updatedHistory.slice(0, 10); // Keep only latest 10
      });
      
      // Also fetch the latest data from backend to ensure consistency
      fetchHistory();
      
    } catch (error) {
      setError(error.response?.data?.detail || 'Inspection failed');
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getDefectTypeColor = (defectType) => {
    const colors = {
      'scratch': 'bg-red-500',
      'missing_component': 'bg-orange-500',
      'misalignment': 'bg-yellow-500',
      'short_circuit': 'bg-purple-500',
      'broken_trace': 'bg-green-500',
      'contamination': 'bg-blue-500',
      'solder_defect': 'bg-pink-500',
      'component_damage': 'bg-indigo-500',
      'wrong_component': 'bg-cyan-500',
      'polarity_error': 'bg-red-600'
    };
    return colors[defectType] || 'bg-gray-500';
  };

  const getSeverityColor = (severity) => {
    const colors = {
      'LOW': 'text-yellow-600 bg-yellow-100',
      'MEDIUM': 'text-orange-600 bg-orange-100',
      'HIGH': 'text-red-600 bg-red-100',
      'CRITICAL': 'text-red-800 bg-red-200'
    };
    return colors[severity] || 'text-gray-600 bg-gray-100';
  };

  const getStateColor = (state) => {
    const colors = {
      'idle': 'text-gray-600 bg-gray-100',
      'waiting_for_pcb': 'text-blue-600 bg-blue-100',
      'capturing': 'text-yellow-600 bg-yellow-100',
      'inspecting': 'text-purple-600 bg-purple-100',
      'showing_results': 'text-green-600 bg-green-100',
      'waiting_for_removal': 'text-orange-600 bg-orange-100',
      'error': 'text-red-600 bg-red-100'
    };
    return colors[state] || 'text-gray-600 bg-gray-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="h-10 w-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="ml-4">
                <h1 className="text-2xl font-bold text-gray-900">Industrial PCB AOI System</h1>
                <p className="text-sm text-gray-600">Advanced Automated Optical Inspection</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Mode Toggle */}
              <div className="flex bg-gray-100 rounded-lg p-1 shadow-sm">
                <button
                  onClick={() => setRealtimeMode(false)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    !realtimeMode 
                      ? 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Manual Mode
                </button>
                <button
                  onClick={() => setRealtimeMode(true)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    realtimeMode 
                      ? 'bg-white text-blue-600 shadow-sm ring-1 ring-blue-200' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Real-time Mode
                </button>
              </div>
              
              {/* Model Status */}
              <div className={`inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium border ${
                modelStatus.trained 
                  ? 'bg-green-50 text-green-700 border-green-200' 
                  : 'bg-red-50 text-red-700 border-red-200'
              }`}>
                <svg className={`w-4 h-4 mr-1 ${modelStatus.trained ? 'text-green-500' : 'text-red-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {modelStatus.trained ? 'Model Ready' : 'Model Not Trained'}
              </div>
              
              {/* Train Model Button */}
              <button
                onClick={trainModel}
                disabled={modelStatus.training}
                className="inline-flex items-center bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white px-4 py-2 rounded-lg font-medium transition-all duration-200 shadow-sm hover:shadow-md disabled:shadow-none"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                {modelStatus.training ? 'Training...' : 'Train Model'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="ml-3 flex-1">
                <p className="text-sm text-red-800">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-3 text-red-400 hover:text-red-600"
              >
                <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Manual Mode */}
        {!realtimeMode && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Upload and Inspection */}
            <div className="lg:col-span-2 space-y-6">
              {/* File Upload */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-xl font-semibold text-gray-900">PCB Image Upload</h2>
                  <p className="text-sm text-gray-600 mt-1">Upload a PCB image for automated inspection</p>
                </div>
                
                <div className="p-6 space-y-4">
                  {/* File Upload Area */}
                  <div
                    className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                      dragActive 
                        ? 'border-indigo-500 bg-indigo-50' 
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                  >
                    {selectedFile ? (
                      <div className="space-y-4">
                        <div className="mx-auto h-16 w-16 bg-green-100 rounded-full flex items-center justify-center">
                          <svg className="h-8 w-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-lg font-medium text-gray-900">{selectedFile.name}</p>
                          <p className="text-sm text-gray-500">{Math.round(selectedFile.size / 1024)} KB</p>
                        </div>
                        <div className="flex justify-center space-x-4">
                          <button
                            onClick={inspectPCB}
                            disabled={loading || !modelStatus.trained || (datasetType === 'marked' && !imageName)}
                            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                            title={datasetType === 'marked' && !imageName ? 'Please enter an image name' : ''}
                          >
                            {loading ? 'Inspecting...' : 'Inspect PCB'}
                          </button>
                          <button
                            onClick={() => {
                              setSelectedFile(null);
                              setInspectionResults(null);
                              setError(null);
                            }}
                            className="bg-gray-300 hover:bg-gray-400 text-gray-700 px-6 py-2 rounded-lg font-medium transition-colors"
                          >
                            Clear
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="mx-auto h-16 w-16 bg-gray-100 rounded-full flex items-center justify-center">
                          <svg className="h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                        </div>
                        <div>
                          <p className="text-lg font-medium text-gray-900">Drop PCB image here</p>
                          <p className="text-sm text-gray-500">or click to browse</p>
                        </div>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleFileSelect}
                          className="hidden"
                          id="file-upload"
                        />
                        <label
                          htmlFor="file-upload"
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 cursor-pointer transition-colors"
                        >
                          Browse Files
                        </label>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              { 
                      <a
                        href={`${API_BASE_URL}/api/pcb/inspections.csv`}
                        className="inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-900 text-white hover:bg-gray-800 transition-all duration-200 shadow-sm hover:shadow-md"
                        style={{ marginLeft: '12px' }}
                      >
                        <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Download CSV
                      </a>
         }
              {inspectionResults && (
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <h2 className="text-xl font-semibold text-gray-900">Inspection Results</h2>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        inspectionResults.is_defective 
                          ? 'bg-red-100 text-red-800' 
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {inspectionResults.is_defective ? '❌ DEFECTIVE' : '✅ GOOD'}
                      </div>
                    </div>
                  </div>

                  {defectiveImage && (
                       <div className="mt-4">
                         <h3 className="text-lg font-bold text-red-600">Defective Image:</h3>
                         <img
                           src={defectiveImage}
                           alt="Defective PCB"
                           className="mt-2 border-2 border-red-500 rounded-lg max-w-full"
                         />
                       </div>
                     )}                     
                  
                  <div className="p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Analysis Summary</h3>
                        <div className="space-y-3">
                          {/* Dataset Type Info */}
                          <div className="flex justify-between">
                            <span className="text-gray-600">Dataset Type:</span>
                            <span className="font-medium capitalize">{datasetType}</span>
                          </div>
                          
                          {/* Show validation metrics for marked dataset */}
                          {datasetType === 'marked' && inspectionResults.validation_metrics && (
                            <>
                              <div className="pt-2 mt-2 border-t border-gray-200">
                                <h4 className="text-sm font-medium text-gray-700 mb-2">Validation Metrics</h4>
                                <div className="space-y-2 text-sm">
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">Precision:</span>
                                    <span className="font-medium">
                                      {(inspectionResults.validation_metrics.precision * 100).toFixed(1)}%
                                      {inspectionResults.validation_metrics.precision_breakdown && 
                                        ` (${inspectionResults.validation_metrics.precision_breakdown.true_positives} TP / 
                                        ${inspectionResults.validation_metrics.precision_breakdown.true_positives + inspectionResults.validation_metrics.precision_breakdown.false_positives} detections)`}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">Recall:</span>
                                    <span className="font-medium">
                                      {(inspectionResults.validation_metrics.recall * 100).toFixed(1)}%
                                      {inspectionResults.validation_metrics.recall_breakdown && 
                                        ` (${inspectionResults.validation_metrics.recall_breakdown.true_positives} TP / 
                                        ${inspectionResults.validation_metrics.recall_breakdown.true_positives + inspectionResults.validation_metrics.recall_breakdown.false_negatives} actual defects)`}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">F1 Score:</span>
                                    <span className="font-medium">
                                      {(inspectionResults.validation_metrics.f1_score * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">True Positives:</span>
                                    <span className="font-medium">
                                      {inspectionResults.validation_metrics.true_positives}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">False Positives:</span>
                                    <span className="font-medium">
                                      {inspectionResults.validation_metrics.false_positives}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">False Negatives:</span>
                                    <span className="font-medium">
                                      {inspectionResults.validation_metrics.false_negatives}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </>
                          )}
                          
                          {/* Standard metrics */}
                          <div className="pt-2 mt-2 border-t border-gray-200">
                            <div className="space-y-3">
                              <div className="flex justify-between">
                                <span className="text-gray-600">Quality Score:</span>
                                <span className="font-medium">{inspectionResults.quality_score?.toFixed(1) || 'N/A'}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Confidence Score:</span>
                                <span className="font-medium">{(inspectionResults.confidence_score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Severity Level:</span>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(inspectionResults.severity_level)}`}>
                                  {inspectionResults.severity_level || 'NONE'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-600">Defects Found:</span>
                                <span className="font-medium">{inspectionResults.defects.length}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Defect Details</h3>
                        {inspectionResults.defects.length > 0 ? (
                          <div className="space-y-2 max-h-40 overflow-y-auto">
                            {inspectionResults.defects.map((defect, index) => (
                              <div key={index} className="flex items-center space-x-3 p-2 bg-gray-50 rounded-lg">
                                <div className={`w-3 h-3 rounded-full ${getDefectTypeColor(defect.type)}`}></div>
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium text-gray-900 capitalize">
                                    {defect.code || defect.type.replace('_', ' ')}
                                  </p>
                                  <p className="text-xs text-gray-500 truncate">
                                    {defect.description || `Confidence: ${(defect.confidence * 100).toFixed(1)}%`}
                                  </p>
                                  {defect.severity && (
                                    <span className={`inline-block px-1 py-0.5 rounded text-xs font-medium ${getSeverityColor(defect.severity)}`}>
                                      {defect.severity}
                                    </span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-green-600 text-sm">No defects detected ✨</p>
                        )}
                      </div>
                    </div>
                    
                    {/* Visualization */}
                    {inspectionResults.visualization_path && (
                      <div className="mt-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Industrial Visual Analysis</h3>
                        <div className="border border-gray-200 rounded-lg overflow-hidden bg-black">
                          <img
                            src={`${API_BASE_URL}${inspectionResults.visualization_path}`}
                            alt="PCB Inspection Result"
                            className="w-full h-auto"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column - Stats and History */}
            <div className="space-y-6">
              {/* Statistics */}
              {stats && (
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-xl font-semibold text-gray-900">Statistics</h2>
                  </div>
                  
                  <div className="p-6">
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">{stats.total_inspections}</p>
                        <p className="text-sm text-gray-600">Total Inspections</p>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <p className="text-2xl font-bold text-green-600">{stats.good_count}</p>
                        <p className="text-sm text-gray-600">Good PCBs</p>
                      </div>
                      <div className="text-center p-4 bg-red-50 rounded-lg">
                        <p className="text-2xl font-bold text-red-600">{stats.defective_count}</p>
                        <p className="text-sm text-gray-600">Defective PCBs</p>
                      </div>
                      <div className="text-center p-4 bg-yellow-50 rounded-lg">
                        <p className="text-2xl font-bold text-yellow-600">
                          {(stats.defect_rate * 100).toFixed(1)}%
                        </p>
                        <p className="text-sm text-gray-600">Defect Rate</p>
                      </div>
                    </div>
                    
                    {stats.defect_types.length > 0 && (
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-3">Common Defects</h3>
                        <div className="space-y-2">
                          {stats.defect_types.slice(0, 6).map((defectType, index) => (
                            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                              <div className="flex items-center space-x-2">
                                <div className={`w-3 h-3 rounded-full ${getDefectTypeColor(defectType.type)}`}></div>
                                <span className="text-sm font-medium capitalize">
                                  {defectType.type.replace('_', ' ')}
                                </span>
                              </div>
                              <span className="text-sm text-gray-600">{defectType.count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}



              {/* Recent Inspections */}

              {/* Recent Inspections */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                                  <div className="px-6 py-4 border-b border-gray-200">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">Recent Inspections</h2>
                      <p className="text-sm text-gray-600 mt-1">Most recent PCB inspections shown first</p>
                    </div>
                  </div>
                
                <div className="p-4">
                                    {history.length > 0 ? (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {history.slice(0, 10).map((inspection, index) => {
                        console.log(`Inspection ${index}:`, inspection); // Debug each inspection
                        return (
                          <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition-colors">
                            <div className="flex items-center space-x-3">
                              {/* Status Badge */}
                              <div className={`w-3 h-3 rounded-full ${
                                inspection.is_defective ? 'bg-red-500' : 'bg-green-500'
                              }`}></div>
                              
                              {/* Filename */}
                              <div className="flex-1">
                                <p className="text-sm font-medium text-gray-900">
                                  {inspection.filename || inspection.image_name || inspection.id || `Inspection ${index + 1}`}
                                </p>
                                <p className="text-xs text-gray-500">
                                  {formatTimestamp(inspection.timestamp)}
                                </p>
                              </div>
                            </div>
                            
                            {/* Status Text */}
                            <div className={`px-2 py-1 rounded text-xs font-medium ${
                              inspection.is_defective ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                            }`}>
                              {inspection.is_defective ? 'DEFECT' : 'GOOD'}
                            </div>
                          </div>
                        );
                      })}
                      
                      {/* Show more indicator */}
                      {history.length > 10 && (
                        <div className="text-center mt-4">
                          <p className="text-sm text-gray-500">
                            Showing 10 of {history.length} recent inspections
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      <p className="mt-2 text-sm text-gray-600">No inspection history yet</p>
                      <p className="text-xs text-gray-500">Upload and inspect a PCB to see results here</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Real-time Mode */}
        {realtimeMode && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left Column - Camera and Controls */}
            <div className="lg:col-span-2 space-y-6">
              {/* Camera Setup */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-xl font-semibold text-gray-900">Camera Setup</h2>
                  <p className="text-sm text-gray-600 mt-1">Connect camera for real-time PCB inspection</p>
                </div>
                
                <div className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Camera Type
                      </label>
                      <select
                        value={selectedCamera}
                        onChange={(e) => setSelectedCamera(e.target.value)}
                        disabled={cameraConnected}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-100"
                      >
                        <option value="opencv">Phone/Webcam (OpenCV)</option>
                        <option value="baumer">Baumer SDK Camera</option>
                      </select>
                    </div>
                    
                    <div className="flex items-end">
                      {!cameraConnected ? (
                        <button
                          onClick={connectCamera}
                          className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Connect Camera
                        </button>
                      ) : (
                        <button
                          onClick={disconnectCamera}
                          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Disconnect Camera
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Camera Status */}
                  <div className="mt-4 p-3 rounded-lg bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className={`w-3 h-3 rounded-full mr-2 ${cameraConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                        <span className="text-sm font-medium text-gray-900">
                          Camera Status: {cameraConnected ? 'Connected' : 'Disconnected'}
                        </span>
                      </div>
                      <div className="flex space-x-1">
                        {cameraConnected && (
                          <>
                            <button
                              onClick={refreshCamera}
                              className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors font-medium"
                              title="Force refresh camera - fixes frozen image"
                            >
                              🔄 Force Refresh
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                    {cameras.length > 0 && (
                      <p className="text-xs text-gray-600 mt-1">
                        Available cameras: {cameras.length}
                      </p>
                    )}
                    
                    {/* Camera Performance Info */}
                    {cameraConnected && (
                      <div className="mt-2 text-xs text-gray-600">
                        <div className="flex justify-between">
                          <span>Preview FPS: ~2</span>
                          <span>Quality: 70%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Resolution: 1280x720</span>
                          <span>Latency: ~500ms</span>
                        </div>
                        <div className="text-center mt-1 text-red-600 font-medium">
                          ⚠️ If image is frozen, click "Force Refresh"
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Camera Preview */}
              {cameraConnected && (
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-xl font-semibold text-gray-900">Live Camera Preview</h2>
                  </div>
                  
                  <div className="p-6">
                    <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                      {cameraPreview ? (
                        <img
                          src={cameraPreview}
                          alt="Camera Preview"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
                            <p className="text-gray-600 mt-2">Loading preview...</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Workflow Controls */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-xl font-semibold text-gray-900">Inspection Workflow</h2>
                  <p className="text-sm text-gray-600 mt-1">Control real-time PCB inspection process</p>
                </div>
                
                <div className="p-6">
                  {/* Current State */}
                  <div className="mb-6 p-4 rounded-lg bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-gray-700">Current State:</span>
                        <span className={`ml-2 px-3 py-1 rounded-full text-sm font-medium ${getStateColor(workflowState.state)}`}>
                          {workflowState.state?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
                        </span>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${workflowState.is_running ? 'bg-green-500' : 'bg-gray-400'}`}></div>
                    </div>
                  </div>

                  {/* Control Buttons */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {!workflowState.is_running ? (
                      <>
                        <button
                          onClick={() => startRealtimeWorkflow(false)}
                          disabled={!cameraConnected || !modelStatus.trained}
                          className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Start Manual Mode
                        </button>
                        <button
                          onClick={() => startRealtimeWorkflow(true)}
                          disabled={!cameraConnected || !modelStatus.trained}
                          className="bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Start Auto Mode
                        </button>
                        <button
                          disabled={true}
                          className="bg-gray-400 text-white px-4 py-2 rounded-lg font-medium cursor-not-allowed"
                        >
                          Workflow Stopped
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={triggerInspection}
                          disabled={workflowState.auto_mode || workflowState.state !== 'waiting_for_pcb'}
                          className="bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Trigger Inspection
                        </button>
                        <button
                          onClick={stopRealtimeWorkflow}
                          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                        >
                          Stop Workflow
                        </button>
                        <div className="flex items-center justify-center px-4 py-2 bg-green-100 text-green-800 rounded-lg">
                          <span className="text-sm font-medium">
                            {workflowState.auto_mode ? 'AUTO MODE' : 'MANUAL MODE'}
                          </span>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Instructions */}
                  {!cameraConnected && (
                    <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <p className="text-sm text-yellow-800">
                        <strong>Step 1:</strong> Connect a camera first to enable real-time inspection.
                      </p>
                    </div>
                  )}
                  {!modelStatus.trained && (
                    <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-sm text-red-800">
                        <strong>Step 2:</strong> Train the model using good PCB samples before starting workflow.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Real-time Stats and Results */}
            <div className="space-y-6">
              {/* Session Statistics */}
              {sessionStats && (
                <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-xl font-semibold text-gray-900">Session Statistics</h2>
                  </div>
                  
                  <div className="p-6">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <p className="text-xl font-bold text-blue-600">{sessionStats.total_inspected || 0}</p>
                        <p className="text-xs text-gray-600">Total Inspected</p>
                      </div>
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <p className="text-xl font-bold text-green-600">{sessionStats.total_good || 0}</p>
                        <p className="text-xs text-gray-600">Good PCBs</p>
                      </div>
                      <div className="text-center p-3 bg-red-50 rounded-lg">
                        <p className="text-xl font-bold text-red-600">{sessionStats.total_defective || 0}</p>
                        <p className="text-xs text-gray-600">Defective PCBs</p>
                      </div>
                      <div className="text-center p-3 bg-yellow-50 rounded-lg">
                        <p className="text-xl font-bold text-yellow-600">
                          {sessionStats.defect_rate?.toFixed(1) || '0.0'}%
                        </p>
                        <p className="text-xs text-gray-600">Defect Rate</p>
                      </div>
                    </div>
                    
                    {sessionStats.avg_inspection_time && (
                      <div className="text-center p-3 bg-purple-50 rounded-lg">
                        <p className="text-lg font-bold text-purple-600">
                          {sessionStats.avg_inspection_time.toFixed(2)}s
                        </p>
                        <p className="text-xs text-gray-600">Avg Inspection Time</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Real-time Results */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-xl font-semibold text-gray-900">Real-time Results</h2>
                </div>
                
                <div className="p-6">
                  {realtimeResults.length > 0 ? (
                    <div className="space-y-3 max-h-80 overflow-y-auto">
                      {realtimeResults.map((result, index) => (
                        <div key={result.result_id || index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-900">
                              {result.result_id || `Result ${index + 1}`}
                            </span>
                            <div className={`px-2 py-1 rounded text-xs font-medium ${
                              result.is_defective 
                                ? 'bg-red-100 text-red-800' 
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {result.is_defective ? 'DEFECTIVE' : 'GOOD'}
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 space-y-1">
                            <div>Quality: {result.quality_score?.toFixed(1) || 'N/A'}%</div>
                            {result.severity_level && (
                              <div>Severity: 
                                <span className={`ml-1 px-1 py-0.5 rounded text-xs ${getSeverityColor(result.severity_level)}`}>
                                  {result.severity_level}
                                </span>
                              </div>
                            )}
                            <div>Time: {result.inspection_time?.toFixed(2) || '0.00'}s</div>
                            <div>Defects: {result.defects?.length || 0}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      <p className="mt-2 text-sm text-gray-600">No inspection results yet</p>
                      <p className="text-xs text-gray-500">Start the workflow to see real-time results</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Quick Guide */}
              <div className="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-xl shadow-lg border border-indigo-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-indigo-200">
                  <h2 className="text-xl font-semibold text-indigo-900">Real-time Guide</h2>
                </div>
                
                <div className="p-6">
                  <div className="space-y-3 text-sm text-indigo-800">
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">1</div>
                      <p>Connect your phone/webcam or Baumer camera via USB</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">2</div>
                      <p>Train the model with good PCB samples</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">3</div>
                      <p>Start workflow in Manual or Auto mode</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">4</div>
                      <p>Place PCB under camera and trigger inspection</p>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">5</div>
                      <p>View real-time results with defect marking</p>
                    </div>
                  </div>
                  
                  <div className="mt-4 p-3 bg-indigo-100 rounded-lg">
                    <p className="text-xs text-indigo-700">
                      <strong>Auto Mode:</strong> Continuous inspection every 15 seconds<br/>
                      <strong>Manual Mode:</strong> Trigger inspection on demand
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;