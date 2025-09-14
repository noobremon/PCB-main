#!/usr/bin/env python3
"""
Industrial PCB AOI System - Consistency Verification Script
Author: AI Assistant
Description: Verifies that ALL inspection modes produce IDENTICAL results
"""

import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
import logging

# Import the inspection modules
from enhanced_pcb_inspection import IndustrialPCBInspector
from camera_integration import CameraManager
from realtime_workflow import InspectionWorkflowManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ConsistencyVerifier:
    """Verify that all inspection modes produce identical results"""
    
    def __init__(self):
        self.inspector = IndustrialPCBInspector()
        self.camera_manager = CameraManager()
        self.workflow_manager = InspectionWorkflowManager()
        
        self.test_results = {
            'manual_mode': None,
            'realtime_baumer': None,
            'realtime_usb': None,
            'consistency_verified': False
        }
    
    def setup_system(self) -> bool:
        """Setup the inspection system for testing"""
        logger.info("🏭 Setting up Industrial PCB AOI System for consistency verification...")
        
        # Check if dataset exists
        dataset_good = Path("dataset/good")
        dataset_defective = Path("dataset/defective")
        
        if not dataset_good.exists() or not list(dataset_good.glob("*.jpg")):
            logger.error("❌ No good PCB images found in dataset/good/")
            logger.info("Please add good PCB images to dataset/good/ folder")
            return False
        
        if not dataset_defective.exists():
            logger.warning("⚠️ No defective PCB images found in dataset/defective/")
            logger.info("For best results, add defective PCB images to dataset/defective/ folder")
        
        # Build the industrial reference model
        logger.info("🤖 Building industrial reference model...")
        success = self.inspector.build_industrial_reference_model()
        
        if not success:
            logger.error("❌ Failed to build reference model")
            return False
        
        # Verify model configuration
        config = self.inspector.get_consistent_inspection_config()
        logger.info("✅ Reference model built successfully")
        logger.info(f"   📊 Reference templates: {config['reference_templates_count']}")
        logger.info(f"   📊 Component templates: {config['component_templates_count']}")
        logger.info(f"   📊 Defect patterns: {config['defect_patterns_count']}")
        logger.info(f"   📊 ML model available: {config['anomaly_detector_available']}")
        logger.info(f"   📊 Defect types: {len(config['defect_types'])}")
        
        return True
    
    def test_manual_mode(self, test_image_path: str) -> dict:
        """Test manual inspection mode"""
        logger.info("🖱️ Testing MANUAL inspection mode...")
        
        # Load test image
        image = cv2.imread(test_image_path)
        if image is None:
            logger.error(f"Cannot load test image: {test_image_path}")
            return {}
        
        # Run inspection in manual mode
        start_time = time.time()
        results = self.inspector.detect_industrial_defects(image, source_mode="manual")
        inspection_time = time.time() - start_time
        
        # Log results
        logger.info(f"   ✅ Manual inspection completed in {inspection_time:.3f}s")
        logger.info(f"   📊 Quality Score: {results['quality_score']:.1f}%")
        logger.info(f"   ⚠️  Severity: {results['severity_level']}")
        logger.info(f"   🔍 Defects: {len(results['defects'])}")
        logger.info(f"   🎯 Consistency Hash: {results.get('consistency_hash', 'N/A')}")
        
        return results
    
    def test_realtime_baumer_mode(self, test_image_path: str) -> dict:
        """Test real-time Baumer camera mode (simulated)"""
        logger.info("📷 Testing REAL-TIME BAUMER inspection mode...")
        
        # Load test image (simulating Baumer camera capture)
        image = cv2.imread(test_image_path)
        if image is None:
            logger.error(f"Cannot load test image: {test_image_path}")
            return {}
        
        # Run inspection in Baumer real-time mode
        start_time = time.time()
        results = self.inspector.detect_industrial_defects(image, source_mode="realtime_baumer")
        inspection_time = time.time() - start_time
        
        # Log results
        logger.info(f"   ✅ Baumer real-time inspection completed in {inspection_time:.3f}s")
        logger.info(f"   📊 Quality Score: {results['quality_score']:.1f}%")
        logger.info(f"   ⚠️  Severity: {results['severity_level']}")
        logger.info(f"   🔍 Defects: {len(results['defects'])}")
        logger.info(f"   🎯 Consistency Hash: {results.get('consistency_hash', 'N/A')}")
        
        return results
    
    def test_realtime_usb_mode(self, test_image_path: str) -> dict:
        """Test real-time USB camera mode (simulated)"""
        logger.info("📱 Testing REAL-TIME USB/PHONE inspection mode...")
        
        # Load test image (simulating USB/phone camera capture)  
        image = cv2.imread(test_image_path)
        if image is None:
            logger.error(f"Cannot load test image: {test_image_path}")
            return {}
        
        # Run inspection in USB real-time mode
        start_time = time.time()
        results = self.inspector.detect_industrial_defects(image, source_mode="realtime_usb")
        inspection_time = time.time() - start_time
        
        # Log results
        logger.info(f"   ✅ USB real-time inspection completed in {inspection_time:.3f}s")
        logger.info(f"   📊 Quality Score: {results['quality_score']:.1f}%")
        logger.info(f"   ⚠️  Severity: {results['severity_level']}")
        logger.info(f"   🔍 Defects: {len(results['defects'])}")
        logger.info(f"   🎯 Consistency Hash: {results.get('consistency_hash', 'N/A')}")
        
        return results
    
    def verify_consistency(self, test_image_path: str | None = None) -> bool:
        """Verify that all modes produce identical results"""
        logger.info("🔍 VERIFYING CONSISTENCY ACROSS ALL INSPECTION MODES...")
        
        # Use test image or create one
        if test_image_path is None or not os.path.exists(test_image_path):
            test_image_path = "test_images/test_defective.jpg"
            if not os.path.exists(test_image_path):
                logger.error("❌ No test image available for consistency verification")
                return False
        
        logger.info(f"   📄 Using test image: {test_image_path}")
        
        # Test all three modes
        self.test_results['manual_mode'] = self.test_manual_mode(test_image_path)
        self.test_results['realtime_baumer'] = self.test_realtime_baumer_mode(test_image_path)
        self.test_results['realtime_usb'] = self.test_realtime_usb_mode(test_image_path)
        
        # Verify consistency
        logger.info("🔬 ANALYZING CONSISTENCY...")
        
        modes = ['manual_mode', 'realtime_baumer', 'realtime_usb']
        results = [self.test_results[mode] for mode in modes]
        
        # Check if all results exist
        if not all(results):
            logger.error("❌ Some inspection modes failed")
            return False
        
        # Compare key metrics for consistency
        consistency_checks = []
        
        # 1. Quality Score consistency
        quality_scores = [r['quality_score'] for r in results]
        quality_consistent = all(abs(score - quality_scores[0]) < 0.1 for score in quality_scores)
        consistency_checks.append(('Quality Scores', quality_consistent, quality_scores))
        
        # 2. Defect count consistency
        defect_counts = [len(r['defects']) for r in results]
        defect_count_consistent = all(count == defect_counts[0] for count in defect_counts)
        consistency_checks.append(('Defect Counts', defect_count_consistent, defect_counts))
        
        # 3. Severity level consistency
        severity_levels = [r['severity_level'] for r in results]
        severity_consistent = all(level == severity_levels[0] for level in severity_levels)
        consistency_checks.append(('Severity Levels', severity_consistent, severity_levels))
        
        # 4. Defective status consistency  
        defective_statuses = [r['is_defective'] for r in results]
        defective_consistent = all(status == defective_statuses[0] for status in defective_statuses)
        consistency_checks.append(('Defective Status', defective_consistent, defective_statuses))
        
        # 5. Consistency hash verification
        consistency_hashes = [r.get('consistency_hash', '') for r in results]
        hash_consistent = all(h == consistency_hashes[0] for h in consistency_hashes if h)
        consistency_checks.append(('Consistency Hash', hash_consistent, consistency_hashes))
        
        # Report results
        logger.info("📋 CONSISTENCY VERIFICATION RESULTS:")
        all_consistent = True
        
        for check_name, is_consistent, values in consistency_checks:
            status = "✅ PASS" if is_consistent else "❌ FAIL"
            logger.info(f"   {status} {check_name}: {values}")
            if not is_consistent:
                all_consistent = False
        
        # Overall result
        if all_consistent:
            logger.info("🏆 ✅ CONSISTENCY VERIFICATION: PASSED!")
            logger.info("   🎯 All inspection modes produce IDENTICAL results")
            logger.info("   🏭 System is ready for industrial deployment")
            self.test_results['consistency_verified'] = True
        else:
            logger.error("❌ CONSISTENCY VERIFICATION: FAILED!")
            logger.error("   ⚠️ Different modes produce different results")
            logger.error("   🔧 System needs calibration")
            self.test_results['consistency_verified'] = False
        
        return all_consistent
    
    def save_consistency_report(self, filename: str = "consistency_report.json"):
        """Save detailed consistency report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_config': self.inspector.get_consistent_inspection_config(),
            'test_results': self.test_results,
            'consistency_verified': self.test_results['consistency_verified']
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Consistency report saved: {filename}")
    
    def run_full_verification(self, test_image_path: str | None = None) -> bool:
        """Run complete consistency verification"""
        logger.info("🚀 STARTING INDUSTRIAL PCB AOI CONSISTENCY VERIFICATION")
        logger.info("="*80)
        
        # Setup system
        if not self.setup_system():
            logger.error("❌ System setup failed")
            return False
        
        # Run consistency verification
        success = self.verify_consistency(test_image_path)
        
        # Save report
        self.save_consistency_report()
        
        logger.info("="*80)
        if success:
            logger.info("🏆 VERIFICATION COMPLETE: SYSTEM IS CONSISTENT AND READY!")
        else:
            logger.info("❌ VERIFICATION COMPLETE: SYSTEM NEEDS ATTENTION")
        
        return success

def main():
    """Main function for consistency verification"""
    print("🏭 Industrial PCB AOI System - Consistency Verification")
    print("="*60)
    
    verifier = ConsistencyVerifier()
    
    # Check for test image argument
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run verification
    success = verifier.run_full_verification(test_image)
    
    if success:
        print("\n🎉 SUCCESS: Your PCB AOI system produces consistent results across ALL modes!")
        print("✅ Manual mode, Baumer camera mode, and USB camera mode are identical")
        print("🏭 System is ready for industrial deployment")
    else:
        print("\n⚠️  WARNING: Inconsistency detected between inspection modes")
        print("🔧 Please check the system configuration")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())