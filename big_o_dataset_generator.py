import json
import random
import textwrap
import uuid

# --- Template Functions for each Big O Complexity Class ---

# O(1) Templates
def generate_o_1_template_1():
    """Accessing an element in a list by index."""
    return textwrap.dedent("""
def get_first_element(data):
    # Accessing by index is a constant time operation
    if data:
        return data[0]
    return None
    """)

def generate_o_1_template_2():
    """Simple arithmetic operation."""
    return textwrap.dedent("""
def perform_simple_calculation(a, b):
    # Basic arithmetic operations are constant time
    x = a * 2
    y = b + 10
    result = x - y
    return result
    """)

# O(log n) Templates
def generate_o_log_n_template_1():
    """Binary search in a sorted list."""
    return textwrap.dedent("""
def binary_search(sorted_list, target):
    # Each step halves the search space
    low = 0
    high = len(sorted_list) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
    """)

def generate_o_log_n_template_2():
    """Finding an element by repeatedly dividing."""
    return textwrap.dedent("""
def find_power_of_two(n):
    # This loop divides n by 2 in each iteration
    i = 1
    while i < n:
        i *= 2
    return i
    """)

# O(n) Templates
def generate_o_n_template_1():
    """Simple for loop over a list."""
    return textwrap.dedent("""
def sum_list_elements(data):
    # The loop runs once for each element in the list
    total = 0
    for item in data:
        total += item
    return total
    """)

def generate_o_n_template_2():
    """While loop to find an element in a list (worst case)."""
    return textwrap.dedent("""
def find_element_in_list(data, target):
    # In the worst case, we have to check every element
    index = 0
    while index < len(data):
        if data[index] == target:
            return index
        index += 1
    return -1
    """)

# O(n log n) Templates
def generate_o_n_log_n_template_1():
    """Merge Sort implementation."""
    return textwrap.dedent("""
def merge_sort(data):
    # Recursively dividing the list is O(log n)
    # Merging the lists is O(n)
    if len(data) > 1:
        mid = len(data) // 2
        left_half = data[:mid]
        right_half = data[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                data[k] = left_half[i]
                i += 1
            else:
                data[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            data[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            data[k] = right_half[j]
            j += 1
            k += 1
    return data
    """)

def generate_o_n_log_n_template_2():
    """Heap Sort (conceptual, using a library for simplicity)."""
    return textwrap.dedent("""
import heapq

def heap_sort(data):
    # Building a heap is O(n)
    # Repeatedly extracting the min element is O(log n), done n times
    heapq.heapify(data)
    sorted_data = []
    while data:
        sorted_data.append(heapq.heappop(data))
    return sorted_data
    """)

# O(n^2) Templates
def generate_o_n_squared_template_1():
    """Nested for loops to find pairs."""
    return textwrap.dedent("""
def find_all_pairs(data):
    # A nested loop where both loops depend on the size of the input
    pairs = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                pairs.append((data[i], data[j]))
    return pairs
    """)

def generate_o_n_squared_template_2():
    """Bubble Sort implementation."""
    return textwrap.dedent("""
def bubble_sort(data):
    # The outer loop runs n times, and the inner loop runs up to n times
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
    """)
    
# O(2^n) Templates
def generate_o_2_n_template_1():
    """Recursive calculation of Fibonacci numbers (inefficient)."""
    return textwrap.dedent("""
def recursive_fibonacci(n):
    # Each call branches into two more calls
    if n <= 1:
        return n
    else:
        return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)
    """)

def generate_o_2_n_template_2():
    """Finding all subsets of a set recursively."""
    return textwrap.dedent("""
def generate_all_subsets(data):
    if not data:
        return [[]]
    
    first = data[0]
    rest = data[1:]
    
    subsets_without_first = generate_all_subsets(rest)
    subsets_with_first = []
    
    for subset in subsets_without_first:
        subsets_with_first.append(subset + [first])
        
    return subsets_without_first + subsets_with_first
    """)


def main():
    """Generates the dataset and saves it to a JSON file."""
    
    templates = {
        "O(1)": [generate_o_1_template_1, generate_o_1_template_2],
        "O(log n)": [generate_o_log_n_template_1, generate_o_log_n_template_2],
        "O(n)": [generate_o_n_template_1, generate_o_n_template_2],
        "O(n log n)": [generate_o_n_log_n_template_1, generate_o_n_log_n_template_2],
        "O(n^2)": [generate_o_n_squared_template_1, generate_o_n_squared_template_2],
        "O(2^n)": [generate_o_2_n_template_1, generate_o_2_n_template_2],
    }

    # Descriptions for each complexity class to add variety
    descriptions = {
        "O(1)": "a constant time algorithm",
        "O(log n)": "a logarithmic time algorithm",
        "O(n)": "a linear time algorithm",
        "O(n log n)": "a log-linear time algorithm",
        "O(n^2)": "a quadratic time algorithm",
        "O(2^n)": "an exponential time algorithm",
    }

    dataset = []
    examples_per_template = 25

    print("Generating Big O dataset...")

    for complexity, template_funcs in templates.items():
        print(f"  Generating examples for {complexity}...")
        for i, template_func in enumerate(template_funcs):
            for j in range(examples_per_template):
                # Generate base code
                code = template_func()

                # Add minor variations (e.g., comments, variable names)
                # This is a simple placeholder for more complex variation logic
                if j % 3 == 0:
                    code = "# A simple implementation\n" + code
                elif j % 3 == 1:
                    code = code.replace("data", "input_list").replace("target", "value")
                
                # Create a unique ID for each snippet
                item_id = f"{complexity.replace(' ', '')}_template{i+1}_{j+1}_{uuid.uuid4().hex[:4]}"

                dataset.append({
                    "id": item_id,
                    "complexity_class": complexity,
                    "code": code,
                    "description": f"This is an example of {descriptions[complexity]}."
                })
    
    # Shuffle the dataset to ensure randomness
    random.shuffle(dataset)

    # Save to file
    output_filename = "big_o_dataset.json"
    with open(output_filename, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSuccessfully generated {len(dataset)} examples.")
    print(f"Dataset saved to {output_filename}")


if __name__ == "__main__":
    main()
