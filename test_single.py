from simple_pcb_inspector import test_on_image

# Test on a single image
input_image = "test_images/PCB_1_MissingHole_01.jpg"
output_image = "results/result_missing_hole.jpg"

try:
    test_on_image(input_image, output_image, debug=True)
    print(f"\nTest complete! Check {output_image} for results.")
except Exception as e:
    print(f"Error: {str(e)}")
