import pytest
import svart
from embeddings_fixture import TEXT_FIXTURE, EMBEDDINGS_FIXTURE, QUERY_FIXTURE


def test_search_returns_correct_data():
    """Test that search returns the correct data."""
    svart_instance = svart.Svart()

    data = [svart.Data(text=TEXT_FIXTURE[i], embedding=EMBEDDINGS_FIXTURE[i])
            for i in range(len(TEXT_FIXTURE))]

    svart_instance.index(data)
    results = svart_instance.search(QUERY_FIXTURE)
    assert results[0].text == TEXT_FIXTURE[2]


def test_correct_indexing():
    """Test that data is correctly indexed and searchable."""
    svart_instance = svart.Svart()

    data = [svart.Data(text=TEXT_FIXTURE[i], embedding=EMBEDDINGS_FIXTURE[i])
            for i in range(len(TEXT_FIXTURE))]
    svart_instance.index(data)

    query_embedding = EMBEDDINGS_FIXTURE[0]
    search_results = svart_instance.search(query_embedding)

    assert len(search_results) == len(data)
    assert search_results[0].text == "How much is a burguer?"



def test_search_results():
    """Test the search functionality."""
    svart_instance = svart.Svart()

    data = [svart.Data(text=TEXT_FIXTURE[i], embedding=EMBEDDINGS_FIXTURE[i])
            for i in range(len(TEXT_FIXTURE))]
    svart_instance.index(data)

    results = svart_instance.search(QUERY_FIXTURE)
    assert len(results) > 0
    assert all(isinstance(node, svart.Node) for node in results)
