import ast
import inspect
import logging
import sys
from textwrap import dedent
from unittest.mock import MagicMock
import builtins
from typing import List, Set, Dict, Any, Optional

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    MAX_CHAIN_LENGTH = 10  # Configurable depth limit for SafeMock attribute chains
    SAFE_IMPORTS = {'sys', 'math'}  # Modules to allow real imports for

# --- SafeCall Wrapper ---
def safe_call_wrapper(func: callable) -> Any:
    """Executes a zero-argument function, returning a SafeMock on exception."""
    try:
        return func()
    except Exception as e:
        logger.debug(f"safe_call_wrapper caught exception: {e}")
        return SafeMock()

# --- SafeMock and TerminalSafeMock Classes ---
class SafeMock:
    def __init__(self, root: bool = False, name: Optional[str] = None, chain: Optional[List[str]] = None):
        self.call_args_list = []
        self.index = 0
        self.dict = {}
        self.terminal = False
        self._is_root = root
        self.name = name
        self._chain = chain or []
    
    def __getattr__(self, name: str) -> 'SafeMock':
        if len(self._chain) >= Config.MAX_CHAIN_LENGTH:
            logger.warning(f"Attribute chain exceeded max length {Config.MAX_CHAIN_LENGTH}: {self._chain}")
            return self
        new_chain = self._chain + [name]
        return SafeMock(root=self._is_root, name=self.name, chain=new_chain)
    
    def __call__(self, *args, **kwargs) -> 'SafeMock':
        if not self.terminal:
            self.call_args_list.append((self._chain, args, kwargs))
            self.terminal = True
        return self

    def __repr__(self) -> str:
        chain_str = ".".join(self._chain)
        return f"<SafeMock: {self.name}.{chain_str}>" if self.name else f"<SafeMock: {chain_str}>"

    # Operator overloads for safe operations
    def __eq__(self, other): return True
    def __bool__(self): return True
    def __iter__(self): return self
    def __next__(self):
        if self.index < 2:
            self.index += 1
            return self
        raise StopIteration
    def __getitem__(self, key): return self.dict.get(key, self)
    def __setitem__(self, key, value): self.dict[key] = value
    def __gt__(self, other): return True
    def __lt__(self, other): return True
    def __ge__(self, other): return True
    def __le__(self, other): return True
    def __eq__(self, other): return True
    def __ne__(self, other): return True
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index < 2:
            self.index += 1
            return self
        else:
            raise StopIteration
    def __setitem__(self, key, value):
        self.dict[key] = value
    def __getitem__(self, key):
        return self.dict.get(key, self)
    def __delitem__(self, key):
        try:
            del self.dict[key]
        except KeyError:
            pass
    def __contains__(self, item):
        return True
    def __len__(self): return 0
    def __bool__(self): return True
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __complex__(self): return complex(0)
    def __index__(self): return 0
    def __bytes__(self): return b''
    def __format__(self, format_spec): return ''
    def __str__(self): return 'SafeMock'
    def __add__(self, other): return self
    def __sub__(self, other): return self
    def __mul__(self, other): return self
    def __truediv__(self, other): return self
    def __floordiv__(self, other): return self
    def __mod__(self, other): return self
    def __divmod__(self, other): return (self, self)
    def __pow__(self, other, modulo=None): return self
    def __radd__(self, other): return self
    def __rsub__(self, other): return self
    def __rmul__(self, other): return self
    def __rtruediv__(self, other): return self
    def __rfloordiv__(self, other): return self
    def __rmod__(self, other): return self
    def __rdivmod__(self, other): return (self, self)
    def __rpow__(self, other): return self
    def __iadd__(self, other): return self
    def __isub__(self, other): return self
    def __imul__(self, other): return self
    def __itruediv__(self, other): return self
    def __ifloordiv__(self, other): return self
    def __imod__(self, other): return self
    def __ipow__(self, other): return self
    def __and__(self, other): return self
    def __or__(self, other): return self
    def __xor__(self, other): return self
    def __lshift__(self, other): return self
    def __rshift__(self, other): return self
    def __rand__(self, other): return self
    def __ror__(self, other): return self
    def __rxor__(self, other): return self
    def __rlshift__(self, other): return self
    def __rrshift__(self, other): return self
    def __iand__(self, other): return self
    def __ior__(self, other): return self
    def __ixor__(self, other): return self
    def __ilshift__(self, other): return self
    def __irshift__(self, other): return self
    def __neg__(self): return self
    def __pos__(self): return self
    def __invert__(self): return self
    def __matmul__(self, other): return self
    def __rmatmul__(self, other): return self
    def __imatmul__(self, other): return self
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): return True
    def __delattr__(self, name): pass
    def __dir__(self): return []
    def __round__(self, ndigits=None): return self
    def __trunc__(self): return self
    def __floor__(self): return self
    def __ceil__(self): return self
    def __hash__(self): return hash('hash')

class TerminalSafeMock(SafeMock):
    """A SafeMock that does not record further calls."""
    def __init__(self):
        super().__init__()
        self.terminal = True
    
    def __call__(self, *args, **kwargs) -> 'TerminalSafeMock':
        return self

# --- Safe Execution Namespace ---
class SafeExecutionNamespace(dict):
    _reserved = set([attr for attr in dir(builtins) if isinstance(getattr(builtins, attr), type) and issubclass(getattr(builtins, attr), BaseException)])
    
    def __missing__(self, key: str) -> Any:
        if key in self._reserved:
            return getattr(builtins, key)
        new_mock = SafeMock(name=key)
        self[key] = new_mock
        return new_mock

def create_mock_namespace() -> SafeExecutionNamespace:
    """Creates a namespace with SafeMock for undefined names."""
    mock_registry = []  # Reset per script execution
    
    def create_and_register_mock(func_name: str):
        def wrapper(*args, **kwargs):
            remove_self = kwargs.pop("__remove_self", False)
            caller_name = "direct_call"
            if remove_self and args:
                caller = args[0]
                args = args[1:]
                caller_name = getattr(caller, 'name', "unknown")
            new_mock = SafeMock(name=func_name)
            new_mock.caller_name = caller_name
            try:
                new_mock(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Mock call failed for {func_name}: {e}")
            mock_registry.append((func_name, new_mock))
            return new_mock
        return wrapper

    initial_namespace = {
        'mock_registry': mock_registry,
        'create_and_register_mock': create_and_register_mock,
        'MagicMock': MagicMock,
        'SafeMock': SafeMock,
        'TerminalSafeMock': TerminalSafeMock,
        'safe_call_wrapper': safe_call_wrapper,
        '__builtins__': builtins.__dict__
    }
    return SafeExecutionNamespace(initial_namespace)

# --- AST Transformer ---
class ASTTransformer(ast.NodeTransformer):
    def __init__(self, target_functions: Set[str], defined_names: Set[str]):
        self.target_functions = target_functions
        self.defined_names = defined_names

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'safe_call_wrapper':
            return node
        # Intercept method calls
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.target_functions:
            new_func = ast.Call(
                func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                args=[ast.Constant(value=node.func.attr)],
                keywords=[]
            )
            new_args = [node.func.value] + node.args
            new_keywords = node.keywords + [ast.keyword(arg="__remove_self", value=ast.Constant(value=True))]
            return ast.Call(func=new_func, args=new_args, keywords=new_keywords)
        # Intercept function calls
        if isinstance(node.func, ast.Name) and node.func.id in self.target_functions:
            return ast.Call(
                func=ast.Call(
                    func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                    args=[ast.Constant(value=node.func.id)],
                    keywords=[]
                ),
                args=node.args,
                keywords=node.keywords
            )
        # Wrap other calls
        lambda_node = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=node
        )
        return ast.Call(func=ast.Name(id='safe_call_wrapper', ctx=ast.Load()), args=[lambda_node], keywords=[])

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return self._handle_import(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        return self._handle_import(node)

    def _handle_import(self, node: ast.Import | ast.ImportFrom) -> ast.AST:
        if isinstance(node, ast.ImportFrom) and node.module in Config.SAFE_IMPORTS:
            return node  # Allow safe imports
        new_nodes = []
        for alias in node.names:
            name = alias.asname or alias.name
            new_nodes.append(ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='name', value=ast.Constant(value=name))]
                )
            ))
        return new_nodes if len(new_nodes) > 1 else new_nodes[0]

    def visit_Assign(self, node: ast.Assign) -> ast.Try:
        self.generic_visit(node)
        except_body = [
            ast.Assign(
                targets=[ast.Name(id=target.id, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[ast.keyword(arg='name', value=ast.Constant(value=target.id))]
                )
            ) if isinstance(target, ast.Name) else ast.Pass()
            for target in node.targets
        ]
        return ast.Try(
            body=[node],
            handlers=[ast.ExceptHandler(type=ast.Name(id='Exception', ctx=ast.Load()), body=except_body or [ast.Pass()])],
            orelse=[], finalbody=[]
        )

# --- Helper Functions ---
def collect_defined_names(tree: ast.AST) -> Set[str]:
    """Collects all defined names in the AST."""
    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
    return defined_names

def transform_ast(script: str, target_functions: Set[str]) -> ast.AST:
    """Transforms the AST to handle errors and intercept calls."""
    script = script.replace('\t', '    ')
    try:
        tree = ast.parse(dedent(script))
    except SyntaxError as e:
        logger.error(f"Failed to parse script: {e}")
        raise ValueError(f"Invalid syntax in script: {e}")
    defined_names = collect_defined_names(tree)
    transformer = ASTTransformer(target_functions, defined_names)
    try:
        transformed_tree = transformer.visit(tree)
        ast.fix_missing_locations(transformed_tree)
        return transformed_tree
    except Exception as e:
        logger.error(f"AST transformation failed: {e}")
        raise

def extract_function_parameters(scripts: List[str], function_names: Set[str]) -> List[Dict[str, Any]]:
    """Extracts parameters from function calls in scripts."""
    results = []
    for script in scripts:
        try:
            transformed_ast = transform_ast(script, function_names)
            namespace = create_mock_namespace()
            exec(compile(transformed_ast, filename="<ast>", mode="exec"), namespace)
            for func_name, mock in namespace.get('mock_registry', []):
                if func_name in function_names:
                    for chain, args, kwargs in mock.call_args_list:
                        processed_args = [repr(arg) if isinstance(arg, SafeMock) else arg for arg in args]
                        processed_kwargs = {k: repr(v) if isinstance(v, SafeMock) else v for k, v in kwargs.items()}
                        results.append({
                            'function': func_name,
                            'caller': ".".join(chain) if chain else getattr(mock, 'caller_name', 'direct_call'),
                            'args': processed_args,
                            'kwargs': processed_kwargs
                        })
        except Exception as e:
            logger.error(f"Error processing script: {e}")
    return remove_duplicates(results)

def remove_duplicates(input_list: List[Any]) -> List[Any]:
    """Removes duplicate entries from a list, handling nested structures."""
    def to_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, to_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(to_hashable(e) for e in obj)
        return obj

    seen = set()
    return [item for item in input_list if not (h := to_hashable(item)) in seen and not seen.add(h)]

# --- Example Usage ---
if __name__ == "__main__":
    scripts = [
        "def foo(x): bar(x); bar(42)",
        "bar('test')"
    ]
    target_functions = {"bar"}
    params = extract_function_parameters(scripts, target_functions)
    for call_info in params:
        print(call_info)