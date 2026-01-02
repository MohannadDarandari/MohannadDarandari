# 🔗 Model Context Protocol (MCP) - Complete Guide 2026

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">

**The Standard for AI Agent Communication**

</div>

---

## 🎯 What is MCP?

**Model Context Protocol (MCP)** is an open protocol that standardizes how AI models and applications connect to external data sources and tools. Think of it as **USB-C for AI** - one standard interface for all integrations.

### Why MCP Matters in 2026

- 🔌 **Universal Integration**: Connect any LLM to any tool with standard protocol
- 🚀 **Reduced Complexity**: No custom integrations for each LLM
- 🔐 **Security First**: Built-in authentication and sandboxing
- ⚡ **Performance**: Efficient JSON-RPC 2.0 communication
- 🌐 **Ecosystem**: Growing library of MCP servers for common tasks

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────────┐
│   LLM Client    │ ◄─────► │ MCP Protocol │ ◄─────► │   MCP Server    │
│  (Claude, GPT)  │         │   (JSON-RPC) │         │ (Tools/Resources)│
└─────────────────┘         └──────────────┘         └─────────────────┘
```

### Components

1. **MCP Client**: The AI application (Claude, custom LLM app)
2. **MCP Protocol**: Standard JSON-RPC 2.0 communication layer
3. **MCP Server**: Provides tools, resources, and prompts to clients

---

## 🛠️ Core Concepts

### 1. Resources
**What**: Files, documents, data that AI can access
**Examples**: 
- Database records
- API responses
- File contents
- Web pages

### 2. Tools (Functions)
**What**: Actions AI can perform
**Examples**:
- Execute SQL queries
- Read/write files
- Call external APIs
- Run code

### 3. Prompts
**What**: Pre-configured prompts with context
**Examples**:
- Code review template
- Data analysis workflow
- Customer service responses

---

## 💻 Building MCP Servers

### Basic MCP Server (Python)

```python
from mcp import MCPServer, Tool, Resource
from typing import List, Dict, Any

class DatabaseMCPServer(MCPServer):
    """MCP Server for database operations"""
    
    def __init__(self):
        super().__init__(
            name="database-server",
            version="1.0.0",
            description="Access PostgreSQL database via MCP"
        )
        
    @self.tool(
        name="query_database",
        description="Execute SQL query and return results",
        parameters={
            "query": {"type": "string", "description": "SQL query to execute"},
            "limit": {"type": "integer", "default": 100}
        }
    )
    async def query_database(self, query: str, limit: int = 100) -> List[Dict]:
        """Execute database query with safety checks"""
        # Validate query (prevent DROP, DELETE without WHERE, etc.)
        if not self._is_safe_query(query):
            raise ValueError("Unsafe query detected")
            
        # Execute query
        results = await self.db.execute(query, limit=limit)
        
        return {
            "success": True,
            "rows": results,
            "count": len(results)
        }
    
    @self.resource(
        uri="database://tables/list",
        name="Database Tables",
        description="List all available database tables"
    )
    async def list_tables(self) -> Dict[str, Any]:
        """Return list of database tables with schema"""
        tables = await self.db.get_tables()
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": t.columns,
                    "row_count": t.row_count
                }
                for t in tables
            ]
        }
    
    def _is_safe_query(self, query: str) -> bool:
        """Validate query safety"""
        dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        query_upper = query.upper()
        return not any(word in query_upper for word in dangerous)

# Run server
if __name__ == "__main__":
    server = DatabaseMCPServer()
    server.run(transport="stdio")  # or "sse", "websocket"
```

### File System MCP Server

```python
from pathlib import Path
import json

class FileSystemMCPServer(MCPServer):
    """MCP Server for file operations"""
    
    def __init__(self, base_path: str = "."):
        super().__init__(
            name="filesystem-server",
            version="1.0.0"
        )
        self.base_path = Path(base_path).resolve()
    
    @self.tool(
        name="read_file",
        description="Read contents of a file",
        parameters={
            "path": {"type": "string", "description": "File path to read"}
        }
    )
    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read file with safety checks"""
        full_path = (self.base_path / path).resolve()
        
        # Security: Prevent path traversal
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError("Path outside allowed directory")
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        content = full_path.read_text(encoding='utf-8')
        
        return {
            "path": str(path),
            "content": content,
            "size": len(content),
            "mime_type": self._get_mime_type(full_path)
        }
    
    @self.tool(
        name="write_file",
        description="Write content to a file",
        parameters={
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_dirs": {"type": "boolean", "default": True}
        }
    )
    async def write_file(self, path: str, content: str, 
                         create_dirs: bool = True) -> Dict:
        """Write file with safety checks"""
        full_path = (self.base_path / path).resolve()
        
        # Security checks
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError("Path outside allowed directory")
        
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content, encoding='utf-8')
        
        return {
            "success": True,
            "path": str(path),
            "bytes_written": len(content)
        }
    
    @self.tool(
        name="list_directory",
        description="List files in a directory",
        parameters={"path": {"type": "string", "default": "."}}
    )
    async def list_directory(self, path: str = ".") -> Dict:
        """List directory contents"""
        full_path = (self.base_path / path).resolve()
        
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError("Path outside allowed directory")
        
        if not full_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        items = []
        for item in full_path.iterdir():
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })
        
        return {
            "path": str(path),
            "items": items,
            "count": len(items)
        }
```

---

## 🚀 Advanced MCP Patterns

### 1. Multi-Tool Server

```python
class AnalyticsMCPServer(MCPServer):
    """Server with multiple related tools"""
    
    @self.tool(name="query_analytics")
    async def query_analytics(self, metric: str, 
                             start_date: str, end_date: str):
        """Query analytics data"""
        pass
    
    @self.tool(name="generate_report")
    async def generate_report(self, report_type: str, format: str = "pdf"):
        """Generate analytics report"""
        pass
    
    @self.tool(name="export_data")
    async def export_data(self, query_id: str, format: str = "csv"):
        """Export analytics data"""
        pass
```

### 2. Streaming Responses

```python
@self.tool(name="stream_logs")
async def stream_logs(self, service: str, 
                      follow: bool = False) -> AsyncIterator[str]:
    """Stream logs in real-time"""
    async for log_line in self.log_service.tail(service, follow=follow):
        yield {
            "timestamp": log_line.timestamp,
            "level": log_line.level,
            "message": log_line.message
        }
```

### 3. Authentication & Authorization

```python
class SecureMCPServer(MCPServer):
    
    async def authenticate(self, token: str) -> bool:
        """Verify client authentication"""
        return await self.auth_service.verify_token(token)
    
    async def authorize(self, user_id: str, action: str, 
                        resource: str) -> bool:
        """Check if user can perform action"""
        return await self.auth_service.check_permission(
            user_id, action, resource
        )
    
    @self.tool(name="admin_action", requires_auth=True, 
              required_role="admin")
    async def admin_action(self, context: MCPContext, **kwargs):
        """Tool that requires admin role"""
        if not await self.authorize(context.user_id, "admin", "*"):
            raise PermissionError("Admin role required")
        
        # Perform admin action
        pass
```

---

## 🎯 Popular MCP Servers (2026)

### Database Servers
- `@modelcontextprotocol/server-postgres`
- `@modelcontextprotocol/server-mysql`
- `@modelcontextprotocol/server-mongodb`

### Cloud Storage
- `@modelcontextprotocol/server-aws-s3`
- `@modelcontextprotocol/server-google-drive`
- `@modelcontextprotocol/server-azure-blob`

### Development Tools
- `@modelcontextprotocol/server-github`
- `@modelcontextprotocol/server-gitlab`
- `@modelcontextprotocol/server-docker`

### Web & APIs
- `@modelcontextprotocol/server-puppeteer`
- `@modelcontextprotocol/server-slack`
- `@modelcontextprotocol/server-notion`

---

## 🔐 Security Best Practices

1. **Input Validation**: Always validate and sanitize inputs
2. **Path Traversal**: Prevent access outside allowed directories
3. **SQL Injection**: Use parameterized queries
4. **Rate Limiting**: Prevent abuse
5. **Authentication**: Verify client identity
6. **Authorization**: Check permissions before actions
7. **Sandboxing**: Isolate execution environments
8. **Logging**: Track all operations for audit

---

## 📊 Performance Optimization

```python
# 1. Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
async def cached_resource(uri: str):
    return await expensive_operation(uri)

# 2. Connection Pooling
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# 3. Batch Operations
@self.tool(name="batch_query")
async def batch_query(self, queries: List[str]):
    """Execute multiple queries in parallel"""
    tasks = [self.execute_query(q) for q in queries]
    return await asyncio.gather(*tasks)
```

---

## 🧪 Testing MCP Servers

```python
import pytest
from mcp.client import MCPClient

@pytest.mark.asyncio
async def test_database_query():
    """Test database query tool"""
    client = MCPClient()
    await client.connect("database-server")
    
    result = await client.call_tool(
        "query_database",
        query="SELECT * FROM users LIMIT 5",
        limit=5
    )
    
    assert result["success"] is True
    assert len(result["rows"]) <= 5

@pytest.mark.asyncio
async def test_file_read():
    """Test file reading"""
    client = MCPClient()
    await client.connect("filesystem-server")
    
    result = await client.call_tool(
        "read_file",
        path="README.md"
    )
    
    assert "content" in result
    assert len(result["content"]) > 0
```

---

## 🌟 Real-World Use Cases

### 1. Code Assistant
- Read project files
- Analyze code structure
- Suggest improvements
- Generate documentation

### 2. Data Analysis Agent
- Query databases
- Generate visualizations
- Export reports
- Schedule analysis

### 3. DevOps Automation
- Deploy applications
- Monitor services
- Manage infrastructure
- Handle incidents

### 4. Customer Support Bot
- Access knowledge base
- Query user data
- Create tickets
- Generate responses

---

## 📚 Resources

- **Official Spec**: https://spec.modelcontextprotocol.io
- **GitHub**: https://github.com/modelcontextprotocol
- **Community Servers**: https://github.com/modelcontextprotocol/servers
- **Documentation**: https://modelcontextprotocol.io/docs

---

*Building the standard for AI tool integration* 🚀
