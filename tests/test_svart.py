import pytest
import svart


def test_data_creation():
    """Test creating an instance of Data."""
    assert hasattr(svart, 'Data'), "Data class is not available in svart module"
    assert 'text' in dir(svart.Data), "Data class does not have a 'text' attribute"

    data_instance = svart.Data(text="example", embedding=[0.1, 0.2, 0.3])
    assert data_instance.text == "example"
    # ... (rest of your test code)



# def test_data_creation():
#     """Test creating an instance of Data."""
#     data_instance = svart.Data(text="example", embedding=[0.1, 0.2, 0.3])
#     assert data_instance.text == "example"
#     assert data_instance.embedding == [0.1, 0.2, 0.3]

def test_svart_new():
    """Test creating an instance of Svart."""
    svart_instance = svart.Svart()
    assert svart_instance is not None

def test_index_and_search():
    """Test indexing and searching functionality of Svart."""
    svart_instance = svart.Svart()
    data_instance = svart.Data(text="example", embedding=[0.1] * 768)
    
    svart_instance.index([data_instance])
    
    # Assuming your search method returns a list of Data instances
    results = svart_instance.search([0.1] * 768)
    assert len(results) > 0
    assert results[0].text == "example"

# Additional tests can be written for other functionalities
