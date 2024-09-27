# Self-Organizing Map (SOM) for Customer Segmentation

This project implements a Self-Organizing Map (SOM) for customer segmentation using Python. It generates a synthetic dataset of customer information and visualizes the segmentation results.

## Software Requirements

- Python 3.7 or higher
- pip (Python package installer)

### Required Python Libraries:
- numpy
- matplotlib
- scikit-learn
- pandas (for optional table output)

## Hardware Requirements

- CPU: Any modern multi-core processor (Intel i3/i5/i7 or AMD equivalent)
- RAM: Minimum 4GB, 8GB or more recommended
- Storage: At least 100MB of free disk space
- Graphics: Basic integrated graphics is sufficient


## Execution Instructions

1. Save the provided Python script as `som_customer_segmentation.py` in your desired directory.

2. Open a terminal or command prompt.

3. Navigate to the directory containing the script:
   ```
   cd path/to/script/directory
   ```

4. Run the script:
   ```
   python som_customer_segmentation.py
   ```

5. The program will execute and display a plot showing the customer segmentation results.

6. If you've added the table output code, it will print a table of sample customer segments in the console and save it as `customer_segments.csv` in the same directory.

## Expected Output

1. A matplotlib window will open, showing a 10x10 grid representing the SOM. Each point on this grid represents a cluster of customers, with colors indicating the normalized age (blue for younger, red for older).

2. In the console, you should see output :

   ```
   Sample Customer Segments:
   Customer   Age   Income      Spending    Segment
   1          47    $123456     $34567      (3, 7)
   2          62    $78901      $12345      (8, 2)
   3          33    $45678      $23456      (1, 9)
   4          55    $98765      $45678      (5, 4)
   5          40    $67890      $56789      (2, 6)
   ```




