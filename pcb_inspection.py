"""
Automated Optical Inspection (AOI) System for PCB Defect Detection
Author: AI Assistant
Description: Advanced PCB inspection system using OpenCV for defect detection and classification
"""

import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, measure, morphology
from sklearn.cluster import DBSCAN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCBInspector:
    """
    Advanced PCB Inspection System for Automated Optical Inspection (AOI)
    """
    
    def __init__(self, dataset_path: str = "dataset", test_path: str = "test_images"):
        """
        Initialize PCB Inspector
        
        Args:
            dataset_path: Path to training dataset directory
            test_path: Path to test images directory
        """
        self.dataset_path = Path(dataset_path)
        self.test_path = Path(test_path)
        self.good_path = self.dataset_path / "good"
        self.defective_path = self.dataset_path / "defective"
        
        # Initialize templates and baselines
        self.reference_templates = []
        self.defect_patterns = []
        self.baseline_features = {}
        
        # Detection parameters
        self.detection_params = {
            'blur_kernel': (5, 5),
            'threshold_binary': 127,
            'morph_kernel_size': (3, 3),
            'contour_min_area': 50,
            'edge_threshold_low': 50,
            'edge_threshold_high': 150,
            'template_match_threshold': 0.8,
            'defect_cluster_eps': 10,
            'defect_cluster_min_samples': 3
        }
        
        # Defect types classification
        self.defect_types = {
            'scratch': {'color': (0, 0, 255), 'description': 'Surface scratch detected'},
            'missing_component': {'color': (255, 0, 0), 'description': 'Missing component detected'},
            'misalignment': {'color': (0, 255, 255), 'description': 'Component misalignment detected'},
            'short_circuit': {'color': (255, 0, 255), 'description': 'Potential short circuit detected'},
            'broken_trace': {'color': (0, 255, 0), 'description': 'Broken trace detected'},
            'contamination': {'color': (255, 165, 0), 'description': 'Surface contamination detected'}
        }
        
        logger.info("PCB Inspector initialized successfully")

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Advanced image preprocessing pipeline
        
        Args:
            image: Input PCB image
            
        Returns:
            Dictionary containing various preprocessed versions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 
                         self.detection_params['edge_threshold_low'], 
                         self.detection_params['edge_threshold_high'])
        
        # Binary thresholding
        _, binary = cv2.threshold(enhanced, 
                                self.detection_params['threshold_binary'], 
                                255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                         self.detection_params['morph_kernel_size'])
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            'original': image,
            'gray': gray,
            'denoised': denoised,
            'enhanced': enhanced,
            'edges': edges,
            'binary': binary,
            'cleaned': cleaned
        }

    def extract_features(self, processed_images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Extract comprehensive features from preprocessed images
        
        Args:
            processed_images: Dictionary of preprocessed images
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Basic statistical features
        gray = processed_images['gray']
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['histogram'] = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Edge density
        edges = processed_images['edges']
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Contour features
        contours, _ = cv2.findContours(processed_images['binary'], 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)
        features['total_contour_area'] = sum(cv2.contourArea(cnt) for cnt in contours)
        
        # Texture features using Local Binary Patterns
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        features['lbp_histogram'] = np.histogram(lbp, bins=10)[0]
        
        # Component detection features
        features['component_count'] = self._count_components(processed_images['cleaned'])
        
        return features

    def _count_components(self, binary_image: np.ndarray) -> int:
        """
        Count electronic components in the image
        
        Args:
            binary_image: Binary image
            
        Returns:
            Number of detected components
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        components = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.detection_params['contour_min_area']:
                # Check if contour resembles a component (rectangular-ish)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratios
                    components += 1
        
        return components

    def build_reference_model(self) -> bool:
        """
        Build reference model from good PCB images
        
        Returns:
            True if successful, False otherwise
        """
        try:
            good_images = glob.glob(str(self.good_path / "*.jpg")) + \
                         glob.glob(str(self.good_path / "*.png")) + \
                         glob.glob(str(self.good_path / "*.bmp"))
            
            if not good_images:
                logger.warning("No good PCB images found for training")
                return False
            
            logger.info(f"Building reference model from {len(good_images)} good PCB images")
            
            reference_features = []
            
            for img_path in good_images:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                processed = self.preprocess_image(image)
                features = self.extract_features(processed)
                reference_features.append(features)
                
                # Store first image as primary template
                if not self.reference_templates:
                    self.reference_templates.append(processed['enhanced'])
            
            # Calculate baseline statistics
            if reference_features:
                self.baseline_features = self._calculate_baseline_stats(reference_features)
                logger.info("Reference model built successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error building reference model: {e}")
            return False

    def _calculate_baseline_stats(self, features_list: List[Dict]) -> Dict[str, Any]:
        """
        Calculate baseline statistics from good PCB features
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Baseline statistics
        """
        baseline = {}
        
        # Calculate mean and std for numerical features
        numerical_features = ['mean_intensity', 'std_intensity', 'edge_density', 
                            'contour_count', 'total_contour_area', 'component_count']
        
        for feature in numerical_features:
            values = [f[feature] for f in features_list if feature in f]
            if values:
                baseline[f'{feature}_mean'] = np.mean(values)
                baseline[f'{feature}_std'] = np.std(values)
        
        # Average histogram
        histograms = [f['histogram'] for f in features_list if 'histogram' in f]
        if histograms:
            baseline['avg_histogram'] = np.mean(histograms, axis=0)
        
        return baseline

    def detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main defect detection function
        
        Args:
            image: Input PCB image to inspect
            
        Returns:
            Detection results with defects and their locations
        """
        logger.info("Starting defect detection...")
        
        # Preprocess image
        processed = self.preprocess_image(image)
        features = self.extract_features(processed)
        
        # Initialize results
        results = {
            'is_defective': False,
            'defects': [],
            'confidence_score': 0.0,
            'processed_images': processed,
            'features': features
        }
        
        # Template matching defect detection
        template_defects = self._detect_template_defects(processed)
        
        # Difference-based defect detection
        difference_defects = self._detect_difference_defects(processed)
        
        # Contour-based defect detection
        contour_defects = self._detect_contour_defects(processed)
        
        # Combine all defects
        all_defects = template_defects + difference_defects + contour_defects
        
        # Remove duplicate detections
        unique_defects = self._remove_duplicate_defects(all_defects)
        
        # Classify defects
        classified_defects = self._classify_defects(unique_defects, processed)
        
        # Calculate overall confidence
        if classified_defects:
            results['is_defective'] = True
            results['defects'] = classified_defects
            results['confidence_score'] = np.mean([d['confidence'] for d in classified_defects])
        
        logger.info(f"Detection complete. Found {len(classified_defects)} defects")
        return results

    def _detect_template_defects(self, processed: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Detect defects using template matching
        
        Args:
            processed: Processed images dictionary
            
        Returns:
            List of template-based defects
        """
        defects = []
        
        if not self.reference_templates:
            return defects
        
        current_image = processed['enhanced']
        
        for template in self.reference_templates:
            # Resize template to match current image if needed
            if template.shape != current_image.shape:
                template = cv2.resize(template, 
                                    (current_image.shape[1], current_image.shape[0]))
            
            # Calculate difference
            diff = cv2.absdiff(current_image, template)
            
            # Threshold difference
            _, thresh_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours in difference
            contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.detection_params['contour_min_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'type': 'template_difference',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(area / 1000.0, 1.0)
                    })
        
        return defects

    def _detect_difference_defects(self, processed: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Detect defects using statistical difference analysis
        
        Args:
            processed: Processed images dictionary
            
        Returns:
            List of difference-based defects
        """
        defects = []
        
        if not self.baseline_features:
            return defects
        
        features = self.extract_features(processed)
        
        # Check feature deviations
        numerical_features = ['mean_intensity', 'std_intensity', 'edge_density', 
                            'contour_count', 'component_count']
        
        significant_deviations = []
        
        for feature in numerical_features:
            if feature in features and f'{feature}_mean' in self.baseline_features:
                current_value = features[feature]
                baseline_mean = self.baseline_features[f'{feature}_mean']
                baseline_std = self.baseline_features[f'{feature}_std']
                
                # Calculate z-score
                if baseline_std > 0:
                    z_score = abs(current_value - baseline_mean) / baseline_std
                    if z_score > 2.0:  # Significant deviation
                        significant_deviations.append({
                            'feature': feature,
                            'z_score': z_score,
                            'current': current_value,
                            'baseline': baseline_mean
                        })
        
        # If significant deviations found, mark as potential defect areas
        if significant_deviations:
            # Use edge detection to find anomalous regions
            edges = processed['edges']
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area and take largest ones as potential defects
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.detection_params['contour_min_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'type': 'statistical_anomaly',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(len(significant_deviations) / 3.0, 1.0),
                        'deviations': significant_deviations
                    })
        
        return defects

    def _detect_contour_defects(self, processed: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Detect defects using contour analysis
        
        Args:
            processed: Processed images dictionary
            
        Returns:
            List of contour-based defects
        """
        defects = []
        
        # Use cleaned binary image for contour detection
        binary = processed['cleaned']
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.detection_params['contour_min_area']:
                # Analyze contour properties
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Calculate circularity (4π * area / perimeter²)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Detect anomalous shapes (potential defects)
                    is_anomalous = False
                    anomaly_type = 'unknown'
                    
                    # Very elongated shapes might be scratches
                    if aspect_ratio > 5 or aspect_ratio < 0.2:
                        is_anomalous = True
                        anomaly_type = 'elongated_shape'
                    
                    # Very irregular shapes might be contamination
                    if circularity < 0.3:
                        is_anomalous = True
                        anomaly_type = 'irregular_shape'
                    
                    # Very large shapes might be missing components
                    if area > 5000:
                        is_anomalous = True
                        anomaly_type = 'large_area'
                    
                    if is_anomalous:
                        defects.append({
                            'type': 'contour_anomaly',
                            'subtype': anomaly_type,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'circularity': circularity,
                            'aspect_ratio': aspect_ratio,
                            'confidence': min(area / 2000.0, 1.0)
                        })
        
        return defects

    def _remove_duplicate_defects(self, defects: List[Dict]) -> List[Dict]:
        """
        Remove duplicate/overlapping defects
        
        Args:
            defects: List of detected defects
            
        Returns:
            List of unique defects
        """
        if not defects:
            return defects
        
        # Extract bounding box centers for clustering
        centers = []
        for defect in defects:
            x, y, w, h = defect['bbox']
            centers.append([x + w/2, y + h/2])
        
        if len(centers) < 2:
            return defects
        
        # Use DBSCAN clustering to group nearby detections
        clustering = DBSCAN(eps=self.detection_params['defect_cluster_eps'], 
                          min_samples=1).fit(centers)
        
        # Keep the highest confidence defect from each cluster
        unique_defects = []
        for cluster_id in set(clustering.labels_):
            cluster_defects = [defects[i] for i, label in enumerate(clustering.labels_) 
                             if label == cluster_id]
            
            # Keep defect with highest confidence
            best_defect = max(cluster_defects, key=lambda d: d['confidence'])
            unique_defects.append(best_defect)
        
        return unique_defects

    def _classify_defects(self, defects: List[Dict], processed: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Classify defects into specific categories
        
        Args:
            defects: List of detected defects
            processed: Processed images dictionary
            
        Returns:
            List of classified defects
        """
        classified = []
        
        for defect in defects:
            x, y, w, h = defect['bbox']
            
            # Extract defect region
            gray = processed['gray']
            defect_region = gray[y:y+h, x:x+w]
            
            if defect_region.size == 0:
                continue
            
            # Classify based on shape and appearance characteristics
            defect_type = 'unknown'
            
            # Scratch detection - long, thin shapes
            if defect.get('aspect_ratio', 1) > 3:
                defect_type = 'scratch'
            
            # Missing component - large, regular shapes
            elif defect.get('area', 0) > 3000 and defect.get('circularity', 0) > 0.5:
                defect_type = 'missing_component'
            
            # Short circuit - small, bright areas
            elif np.mean(defect_region) > 200 and defect.get('area', 0) < 500:
                defect_type = 'short_circuit'
            
            # Contamination - irregular, dark areas
            elif defect.get('circularity', 1) < 0.4 and np.mean(defect_region) < 100:
                defect_type = 'contamination'
            
            # Broken trace - elongated, dark areas
            elif defect.get('aspect_ratio', 1) > 2 and np.mean(defect_region) < 120:
                defect_type = 'broken_trace'
            
            # Misalignment - moderate size, moderate brightness
            else:
                defect_type = 'misalignment'
            
            classified_defect = {
                'type': defect_type,
                'bbox': defect['bbox'],
                'area': defect.get('area', 0),
                'confidence': defect['confidence'],
                'description': self.defect_types.get(defect_type, {}).get('description', 'Unknown defect'),
                'color': self.defect_types.get(defect_type, {}).get('color', (255, 255, 255))
            }
            
            classified.append(classified_defect)
        
        return classified

    def visualize_defects(self, image: np.ndarray, results: Dict[str, Any], 
                         save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected defects on the image
        
        Args:
            image: Original PCB image
            results: Detection results
            save_path: Optional path to save the visualization
            
        Returns:
            Annotated image with defect markings
        """
        annotated = image.copy()
        
        if not results['defects']:
            # Add "GOOD PCB" text
            cv2.putText(annotated, "GOOD PCB", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            return annotated
        
        # Draw defect bounding boxes and labels
        for i, defect in enumerate(results['defects']):
            x, y, w, h = defect['bbox']
            color = defect['color']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Add defect label
            label = f"{defect['type']} ({defect['confidence']:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add overall status
        status = "DEFECTIVE PCB" if results['is_defective'] else "GOOD PCB"
        status_color = (0, 0, 255) if results['is_defective'] else (0, 255, 0)
        
        cv2.putText(annotated, status, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        # Add confidence score
        conf_text = f"Confidence: {results['confidence_score']:.2f}"
        cv2.putText(annotated, conf_text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, annotated)
            logger.info(f"Visualization saved to {save_path}")
        
        return annotated

    def inspect_pcb(self, image_path: str, visualize: bool = True, 
                   save_result: bool = True) -> Dict[str, Any]:
        """
        Complete PCB inspection pipeline
        
        Args:
            image_path: Path to PCB image to inspect
            visualize: Whether to create visualization
            save_result: Whether to save results
            
        Returns:
            Complete inspection results
        """
        logger.info(f"Inspecting PCB: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return {'error': 'Could not load image'}
        
        # Detect defects
        results = self.detect_defects(image)
        
        # Add metadata
        results['image_path'] = image_path
        results['image_shape'] = image.shape
        results['timestamp'] = str(np.datetime64('now'))
        
        # Visualize if requested
        if visualize:
            annotated = self.visualize_defects(image, results)
            results['annotated_image'] = annotated
            
            if save_result:
                # Save annotated image
                base_name = Path(image_path).stem
                save_path = f"inspection_result_{base_name}.jpg"
                cv2.imwrite(save_path, annotated)
                results['visualization_path'] = save_path
        
        # Save results JSON if requested
        if save_result:
            results_for_json = {k: v for k, v in results.items() 
                              if k not in ['processed_images', 'annotated_image']}
            
            json_path = f"inspection_result_{Path(image_path).stem}.json"
            with open(json_path, 'w') as f:
                json.dump(results_for_json, f, indent=2, default=str)
            results['results_path'] = json_path
        
        return results

    def batch_inspect(self, image_dir: str) -> List[Dict[str, Any]]:
        """
        Batch inspection of multiple PCB images
        
        Args:
            image_dir: Directory containing PCB images
            
        Returns:
            List of inspection results
        """
        image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(image_dir, "*.png")) + \
                     glob.glob(os.path.join(image_dir, "*.bmp"))
        
        logger.info(f"Starting batch inspection of {len(image_paths)} images")
        
        results = []
        for image_path in image_paths:
            result = self.inspect_pcb(image_path)
            results.append(result)
        
        # Generate batch summary
        defective_count = sum(1 for r in results if r.get('is_defective', False))
        
        logger.info(f"Batch inspection complete. {defective_count}/{len(results)} PCBs are defective")
        
        return results

def main():
    """
    Main function for standalone usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB Automated Optical Inspection')
    parser.add_argument('--train', action='store_true', help='Build reference model from good PCBs')
    parser.add_argument('--inspect', type=str, help='Inspect single PCB image')
    parser.add_argument('--batch', type=str, help='Batch inspect directory of PCB images')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset directory path')
    parser.add_argument('--test', type=str, default='test_images', help='Test images directory path')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = PCBInspector(dataset_path=args.dataset, test_path=args.test)
    
    if args.train:
        print("Building reference model from good PCB images...")
        success = inspector.build_reference_model()
        if success:
            print("✅ Reference model built successfully!")
        else:
            print("❌ Failed to build reference model. Check if good PCB images exist.")
    
    elif args.inspect:
        print(f"Inspecting PCB: {args.inspect}")
        
        # Build reference model if not exists
        if not inspector.baseline_features:
            print("Building reference model first...")
            inspector.build_reference_model()
        
        result = inspector.inspect_pcb(args.inspect)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            status = "DEFECTIVE" if result['is_defective'] else "GOOD"
            print(f"📋 Inspection Result: {status}")
            print(f"🎯 Confidence: {result['confidence_score']:.2f}")
            
            if result['defects']:
                print(f"🔍 Defects found: {len(result['defects'])}")
                for i, defect in enumerate(result['defects']):
                    print(f"  {i+1}. {defect['type']}: {defect['description']} (conf: {defect['confidence']:.2f})")
    
    elif args.batch:
        print(f"Batch inspecting directory: {args.batch}")
        
        # Build reference model if not exists
        if not inspector.baseline_features:
            print("Building reference model first...")
            inspector.build_reference_model()
        
        results = inspector.batch_inspect(args.batch)
        
        print(f"📊 Batch Inspection Complete:")
        defective_count = sum(1 for r in results if r.get('is_defective', False))
        print(f"   Total PCBs: {len(results)}")
        print(f"   Defective: {defective_count}")
        print(f"   Good: {len(results) - defective_count}")
    
    else:
        print("Please specify --train, --inspect <image>, or --batch <directory>")
        print("Example usage:")
        print("  python pcb_inspection.py --train")
        print("  python pcb_inspection.py --inspect test_images/pcb1.jpg")
        print("  python pcb_inspection.py --batch test_images/")

if __name__ == "__main__":
    main()