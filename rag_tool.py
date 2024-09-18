from langchain.retrievers import ContextualCompressionRetriever, SelfQueryRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.chains.query_constructor.schema import AttributeInfo


def create_retriever(llm, vectorstore):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )


def create_self_query_retriever(llm, vectorstore, metadata_options):
    metadata_field_info = [
        AttributeInfo(
            name="file",
            description=f"The file the information comes from. Options are {metadata_options['files']}",
            type="string",
        ),
        AttributeInfo(
            name="topic",
            description=f"The topic of the information. Options are {metadata_options['topics']}",
            type="string",
        ),
        AttributeInfo(
            name="timeframe",
            description=f"The timeframe associated with the information. The only permissible options are {metadata_options['timeframes']}; you must ONLY use these exact strings with `eq` and MUST NOT attempt to do `gt` or `lt` comparisons. You should use `or` to capture date ranges for a whole year or longer if required.",
            type="string",
        ),
    ]

    document_content_description = "Detailed predictions about developments in AI technology and geopolitics upto 2030."

    return SelfQueryRetriever.from_llm(
        llm.with_config(tags=["self_query_llm"]),
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True,
    )


def create_compression_retriever(llm, base_retriever):
    # compressor = CohereRerank() # TODO need to get Cohere API key
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
