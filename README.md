# AI Foreign Policy Assistant

## Key Features

- **GraphRAG System**: Combines the power of knowledge graphs (via Neo4j) with an LLM to provide advanced retrieval and generation capabilities.
- **Foreign Policy Focus**: Tailored for high-level decision-making in foreign policy, where multiple perspectives, global implications, and rich relationships between nations are essential.
- **Enhanced Contextual Understanding**: Knowledge graphs allow the LLM to understand and analyze more complex relationships and entities, offering more informed and contextual responses compared to traditional RAG.


## System Architecture

1. **Neo4j Knowledge Graph**: Stores structured data related to foreign policy, such as relationships between nations, treaties, trade agreements, historical events, and other global factors.
2. **LLM Integration**: The LLM is augmented with the ability to query the Neo4j knowledge graph, allowing it to retrieve, analyze, and synthesize information from both structured (knowledge graph) and unstructured sources.
3. **GraphRAG Framework**: Utilizes retrieval-augmented generation (RAG) principles, but enhances the process by incorporating graph-based data for more informed responses.
