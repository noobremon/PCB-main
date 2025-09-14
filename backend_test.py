#!/usr/bin/env python3
"""
PCB AOI System Backend API Testing Suite
Tests all backend endpoints for the PCB Automated Optical Inspection system
"""

import requests
import sys
import os
import json
from datetime import datetime
from pathlib import Path

class PCBAPITester:
    def __init__(self, base_url="https://452d2296-eace-4798-8863-6a16c09a1b60.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            'name': name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            response = None
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=60)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                error_msg = f"Unsupported HTTP method: {method}"
                print(f"   Error: {error_msg} ❌")
                self.log_test(name, False, error_msg)
                return False, {}

            if response is None:
                error_msg = "No response received"
                print(f"   Error: {error_msg} ❌")
                self.log_test(name, False, error_msg)
                return False, {}

            success = response.status_code == expected_status
            
            if success:
                print(f"   Status: {response.status_code} ✅")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    self.log_test(name, True)
                    return True, response_data
                except:
                    # For file responses or non-JSON responses
                    self.log_test(name, True, f"Non-JSON response, content-type: {response.headers.get('content-type', 'unknown')}")
                    return True, {}
            else:
                error_msg = f"Expected {expected_status}, got {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'No detail provided')
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - Response: {response.text[:200]}"
                
                print(f"   Status: {response.status_code} ❌")
                self.log_test(name, False, error_msg)
                return False, {}

        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            print(f"   Error: {error_msg} ❌")
            self.log_test(name, False, error_msg)
            return False, {}
        except Exception as e:
            error_msg = f"Request error: {str(e)}"
            print(f"   Error: {error_msg} ❌")
            self.log_test(name, False, error_msg)
            return False, {}

    def test_basic_endpoints(self):
        """Test basic API endpoints"""
        print("\n" + "="*60)
        print("TESTING BASIC API ENDPOINTS")
        print("="*60)
        
        # Test root endpoint
        self.run_test("API Root", "GET", "api/", 200)
        
        # Test status endpoints
        self.run_test("Create Status Check", "POST", "api/status", 200, 
                     data={"client_name": "test_client"})
        
        self.run_test("Get Status Checks", "GET", "api/status", 200)

    def test_pcb_stats_endpoint(self):
        """Test PCB statistics endpoint"""
        print("\n" + "="*60)
        print("TESTING PCB STATISTICS ENDPOINT")
        print("="*60)
        
        success, response = self.run_test("Get PCB Stats", "GET", "api/pcb/stats", 200)
        
        if success and response:
            # Validate response structure
            expected_keys = ['total_inspections', 'defective_count', 'good_count', 
                           'defect_rate', 'defect_types', 'reference_model_available']
            
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                self.log_test("PCB Stats Response Structure", False, 
                            f"Missing keys: {missing_keys}")
            else:
                self.log_test("PCB Stats Response Structure", True)
                print(f"   📊 Total inspections: {response.get('total_inspections', 0)}")
                print(f"   📊 Reference model available: {response.get('reference_model_available', False)}")

    def test_pcb_inspections_endpoint(self):
        """Test PCB inspections history endpoint"""
        print("\n" + "="*60)
        print("TESTING PCB INSPECTIONS HISTORY ENDPOINT")
        print("="*60)
        
        success, response = self.run_test("Get PCB Inspections", "GET", "api/pcb/inspections", 200)
        
        if success:
            if isinstance(response, list):
                self.log_test("PCB Inspections Response Type", True)
                print(f"   📋 Found {len(response)} inspection records")
                
                # If there are inspections, validate structure
                if response:
                    first_inspection = response[0]
                    expected_keys = ['id', 'image_name', 'is_defective', 'confidence_score', 
                                   'defects', 'timestamp']
                    missing_keys = [key for key in expected_keys if key not in first_inspection]
                    if missing_keys:
                        self.log_test("Inspection Record Structure", False, 
                                    f"Missing keys: {missing_keys}")
                    else:
                        self.log_test("Inspection Record Structure", True)
            else:
                self.log_test("PCB Inspections Response Type", False, 
                            "Expected list, got " + str(type(response)))

    def test_pcb_train_endpoint(self):
        """Test PCB model training endpoint"""
        print("\n" + "="*60)
        print("TESTING PCB MODEL TRAINING ENDPOINT")
        print("="*60)
        
        success, response = self.run_test("Train PCB Model", "POST", "api/pcb/train", 200)
        
        if success and response:
            if 'success' in response and 'message' in response:
                self.log_test("Train Response Structure", True)
                print(f"   🎯 Training success: {response.get('success', False)}")
                print(f"   💬 Message: {response.get('message', 'No message')}")
            else:
                self.log_test("Train Response Structure", False, 
                            "Missing 'success' or 'message' in response")

    def test_pcb_inspect_endpoint(self):
        """Test PCB inspection endpoint with sample images"""
        print("\n" + "="*60)
        print("TESTING PCB INSPECTION ENDPOINT")
        print("="*60)
        
        # Test with good PCB image
        good_image_path = "/app/test_images/test_good.jpg"
        if os.path.exists(good_image_path):
            print(f"📁 Testing with good PCB image: {good_image_path}")
            
            with open(good_image_path, 'rb') as f:
                files = {'file': ('test_good.jpg', f, 'image/jpeg')}
                success, response = self.run_test("Inspect Good PCB", "POST", "api/pcb/inspect", 200, files=files)
                
                if success and response:
                    # Validate inspection response structure
                    expected_keys = ['id', 'image_name', 'is_defective', 'confidence_score', 
                                   'defects', 'timestamp', 'visualization_path']
                    missing_keys = [key for key in expected_keys if key not in response]
                    if missing_keys:
                        self.log_test("Good PCB Inspection Response Structure", False, 
                                    f"Missing keys: {missing_keys}")
                    else:
                        self.log_test("Good PCB Inspection Response Structure", True)
                        print(f"   🔍 Is defective: {response.get('is_defective', 'unknown')}")
                        print(f"   🎯 Confidence: {response.get('confidence_score', 0):.3f}")
                        print(f"   🚨 Defects found: {len(response.get('defects', []))}")
                        
                        # Test visualization endpoint if path provided
                        viz_path = response.get('visualization_path')
                        if viz_path:
                            viz_endpoint = viz_path.replace('/api/', 'api/')
                            self.run_test("Get Good PCB Visualization", "GET", viz_endpoint, 200)
        else:
            self.log_test("Good PCB Image File", False, f"File not found: {good_image_path}")
        
        # Test with defective PCB image
        defective_image_path = "/app/test_images/test_defective.jpg"
        if os.path.exists(defective_image_path):
            print(f"\n📁 Testing with defective PCB image: {defective_image_path}")
            
            with open(defective_image_path, 'rb') as f:
                files = {'file': ('test_defective.jpg', f, 'image/jpeg')}
                success, response = self.run_test("Inspect Defective PCB", "POST", "api/pcb/inspect", 200, files=files)
                
                if success and response:
                    # Validate inspection response structure
                    expected_keys = ['id', 'image_name', 'is_defective', 'confidence_score', 
                                   'defects', 'timestamp', 'visualization_path']
                    missing_keys = [key for key in expected_keys if key not in response]
                    if missing_keys:
                        self.log_test("Defective PCB Inspection Response Structure", False, 
                                    f"Missing keys: {missing_keys}")
                    else:
                        self.log_test("Defective PCB Inspection Response Structure", True)
                        print(f"   🔍 Is defective: {response.get('is_defective', 'unknown')}")
                        print(f"   🎯 Confidence: {response.get('confidence_score', 0):.3f}")
                        print(f"   🚨 Defects found: {len(response.get('defects', []))}")
                        
                        # Test visualization endpoint if path provided
                        viz_path = response.get('visualization_path')
                        if viz_path:
                            viz_endpoint = viz_path.replace('/api/', 'api/')
                            self.run_test("Get Defective PCB Visualization", "GET", viz_endpoint, 200)
        else:
            self.log_test("Defective PCB Image File", False, f"File not found: {defective_image_path}")

    def test_error_scenarios(self):
        """Test error handling scenarios"""
        print("\n" + "="*60)
        print("TESTING ERROR SCENARIOS")
        print("="*60)
        
        # Test inspection without file
        self.run_test("Inspect Without File", "POST", "api/pcb/inspect", 422)
        
        # Test non-existent visualization
        self.run_test("Get Non-existent Visualization", "GET", "api/pcb/result/nonexistent.jpg", 404)
        
        # Test invalid file upload
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        success, response = self.run_test("Inspect Invalid File", "POST", "api/pcb/inspect", 400, files=files)

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting PCB AOI System Backend API Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run test suites
        self.test_basic_endpoints()
        self.test_pcb_stats_endpoint()
        self.test_pcb_inspections_endpoint()
        self.test_pcb_train_endpoint()
        self.test_pcb_inspect_endpoint()
        self.test_error_scenarios()
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"📊 Tests run: {self.tests_run}")
        print(f"✅ Tests passed: {self.tests_passed}")
        print(f"❌ Tests failed: {self.tests_run - self.tests_passed}")
        print(f"📈 Success rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "No tests run")
        
        # Show failed tests
        failed_tests = [test for test in self.test_results if not test['success']]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['name']}: {test['details']}")
        
        # Save detailed results
        results_file = f"backend_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'tests_run': self.tests_run,
                    'tests_passed': self.tests_passed,
                    'success_rate': self.tests_passed/self.tests_run if self.tests_run > 0 else 0,
                    'timestamp': datetime.now().isoformat(),
                    'base_url': self.base_url
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return self.tests_passed == self.tests_run

def main():
    """Main function"""
    tester = PCBAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Backend API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check the results above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())