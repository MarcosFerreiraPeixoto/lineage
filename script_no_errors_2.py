import ast
import inspect
from textwrap import dedent
from unittest.mock import MagicMock
import builtins

# --- SafeCall wrapper ---
def safe_call_wrapper(func):
    """Executes the given zero-argument function. If an exception occurs,
    returns a new SafeMock instance instead."""
    try:
        return func()
    except Exception:
        return SafeMock()

# --- SafeMock and TerminalSafeMock definitions ---
class SafeMock:
    def __init__(self, root=False, name=None, chain=None):
        self.call_args_list = []
        self.index = 0
        self.dict = {}
        self.terminal = False
        self._is_root = root
        self.name = name
        # Keep track of attribute chain in a list.
        self._chain = chain or []
        
    def __getattr__(self, name):
        # Limit chain length to avoid runaway recursion.
        if len(self._chain) > 10:
            return self
        new_chain = self._chain + [name]
        return SafeMock(root=self._is_root, name=self.name, chain=new_chain)
    
    def __call__(self, *args, **kwargs):
        if not self.terminal:
            self.call_args_list.append((self._chain, args, kwargs))
            self.terminal = True
        return self

    def __repr__(self):
        chain_str = ".".join(self._chain)
        if self.name:
            return f"<SafeMock: {self.name}.{chain_str}>"
        return f"<SafeMock: {chain_str}>"

    # Basic operator overloads for safe arithmetic, comparisons, etc.
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
    """
    A terminal mock that does not record any further calls.
    """
    def __init__(self):
        super().__init__()
        self.terminal = True

    def __call__(self, *args, **kwargs):
        return self

    def count(self):
        return 1

# --- Custom global namespace that returns SafeMock for missing names ---
class SafeExecutionNamespace(dict):
    _reserved = {"Exception", "BaseException", "ArithmeticError", "LookupError",
                 "AssertionError", "AttributeError", "EOFError", "FloatingPointError",
                 "GeneratorExit", "ImportError", "ModuleNotFoundError", "IndexError",
                 "KeyError", "KeyboardInterrupt", "MemoryError", "NameError", "NotImplementedError",
                 "OSError", "OverflowError", "RecursionError", "ReferenceError", "RuntimeError",
                 "StopIteration", "StopAsyncIteration", "SyntaxError", "IndentationError",
                 "TabError", "SystemError", "SystemExit", "TypeError", "UnboundLocalError",
                 "UnicodeError", "UnicodeEncodeError", "UnicodeDecodeError", "UnicodeTranslateError",
                 "ValueError", "ZeroDivisionError"}
    
    def __missing__(self, key):
        if key in self._reserved:
            return getattr(builtins, key)
        new_mock = SafeMock(name=key)
        self[key] = new_mock
        return new_mock

def create_mock_namespace():
    """Creates a global namespace that supplies SafeMock instances for missing names."""
    mock_registry = []
    
    def create_and_register_mock(func_name):
        def wrapper(*args, **kwargs):
            remove_self = kwargs.pop("__remove_self", False)
            if remove_self and args:
                caller = args[0]
                args = args[1:]
                caller_name = getattr(caller, 'name', "unknown")
            else:
                caller_name = "direct_call"
            new_mock = SafeMock(name=func_name)
            new_mock.caller_name = caller_name
            try:
                new_mock(*args, **kwargs)
            except Exception:
                # If the call fails, continue without breaking.
                pass
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
        'Exception': builtins.Exception,
        'BaseException': builtins.BaseException,
        '__builtins__': builtins.__dict__
    }
    
    return SafeExecutionNamespace(initial_namespace)

# --- Helper: wrap a statement in a try/except block ---
def wrap_stmt_in_try(stmt):
    return ast.Try(
        body=[stmt],
        handlers=[ast.ExceptHandler(
            type=ast.Name(id='Exception', ctx=ast.Load()),
            name=None,
            body=[ast.Pass()]
        )],
        orelse=[], finalbody=[]
    )

# --- AST Transformer ---
class ASTTransformer(ast.NodeTransformer):
    """Transforms the AST to intercept target function/method calls and to wrap
    vulnerable constructs (including assignments, expressions, and control flow)
    so that errors result in SafeMock assignments rather than aborting execution."""
    def __init__(self, target_functions, defined_names):
        self.target_functions = target_functions
        self.defined_names = defined_names

    def visit_Module(self, node):
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        # If this is an __init__ method, wrap each statement in its body in try/except.
        if node.name == '__init__':
            new_body = []
            for stmt in node.body:
                new_body.append(wrap_stmt_in_try(stmt))
            node.body = new_body
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        if getattr(node, '_wrapped_if', False):
            return node
        self.generic_visit(node)
        true_body = node.body if node.body else [ast.Pass()]
        false_body = node.orelse if node.orelse else [ast.Pass()]
        true_try = ast.Try(
            body=true_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()]
            )],
            orelse=[], finalbody=[]
        )
        false_try = ast.Try(
            body=false_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()]
            )],
            orelse=[], finalbody=[]
        )
        node.body = [true_try]
        node.orelse = [false_try]
        node._wrapped_if = True
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        # Wrap expression statements so that errors inside them are caught.
        return ast.copy_location(wrap_stmt_in_try(node), node)

    def visit_For(self, node):
        self.generic_visit(node)
        node.body = [wrap_stmt_in_try(stmt) for stmt in node.body]
        node.orelse = [wrap_stmt_in_try(stmt) for stmt in node.orelse]
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.body = [wrap_stmt_in_try(stmt) for stmt in node.body]
        node.orelse = [wrap_stmt_in_try(stmt) for stmt in node.orelse]
        return node

    def visit_With(self, node):
        self.generic_visit(node)
        node.body = [wrap_stmt_in_try(stmt) for stmt in node.body]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        # Avoid double-wrapping safe_call_wrapper calls.
        if isinstance(node.func, ast.Name) and node.func.id == 'safe_call_wrapper':
            return node
        # Intercept method calls on attributes.
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.target_functions:
            new_func = ast.Call(
                func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                args=[ast.Constant(value=node.func.attr)],
                keywords=[]
            )
            new_args = [node.func.value] + node.args
            new_keywords = node.keywords + [ast.keyword(arg="__remove_self", value=ast.Constant(value=True))]
            new_node = ast.Call(
                func=new_func,
                args=new_args,
                keywords=new_keywords
            )
            return ast.copy_location(new_node, node)
        # Intercept direct function calls.
        if isinstance(node.func, ast.Name) and node.func.id in self.target_functions:
            new_node = ast.Call(
                func=ast.Call(
                    func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                    args=[ast.Constant(value=node.func.id)],
                    keywords=[]
                ),
                args=node.args,
                keywords=node.keywords
            )
            return ast.copy_location(new_node, node)
        # For all other calls, wrap the call in safe_call_wrapper using a lambda.
        lambda_node = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=node
        )
        new_node = ast.Call(
            func=ast.Name(id='safe_call_wrapper', ctx=ast.Load()),
            args=[lambda_node],
            keywords=[]
        )
        return ast.copy_location(new_node, node)

    def visit_Import(self, node):
        return self._mock_import(node)

    def visit_ImportFrom(self, node):
        return self._mock_import(node)

    def _mock_import(self, node):
        new_nodes = []
        for alias in node.names:
            name = alias.asname or alias.name
            new_nodes.append(ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg='name', value=ast.Constant(value=name)),
                        ast.keyword(arg='root', value=ast.Constant(value=True))
                    ]
                )
            ))
        if len(new_nodes) == 1:
            return new_nodes[0]
        return new_nodes

    # Wrap assignment nodes so that if evaluation fails, a SafeMock is assigned.
    def visit_Assign(self, node):
        self.generic_visit(node)
        try_body = [node]
        except_body = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                assign_stmt = ast.Assign(
                    targets=[ast.copy_location(ast.Name(id=target.id, ctx=ast.Store()), target)],
                    value=ast.Call(
                        func=ast.Name(id='SafeMock', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg='name', value=ast.Constant(value=target.id)),
                            ast.keyword(arg='root', value=ast.Constant(value=True))
                        ]
                    )
                )
                except_body.append(assign_stmt)
            else:
                except_body.append(ast.Expr(value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                )))
        if not except_body:
            except_body = [ast.Pass()]
        try_except_node = ast.Try(
            body=try_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=except_body
            )],
            orelse=[], finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        try_body = [node]
        target = node.target
        if isinstance(target, ast.Name):
            except_body = [ast.Assign(
                targets=[ast.copy_location(ast.Name(id=target.id, ctx=ast.Store()), target)],
                value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg='name', value=ast.Constant(value=target.id)),
                        ast.keyword(arg='root', value=ast.Constant(value=True))
                    ]
                )
            )]
        else:
            except_body = [ast.Pass()]
        try_except_node = ast.Try(
            body=try_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=except_body
            )],
            orelse=[], finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        try_body = [node]
        if isinstance(node.target, ast.Name):
            except_body = [ast.Assign(
                targets=[ast.copy_location(ast.Name(id=node.target.id, ctx=ast.Store()), node.target)],
                value=ast.Call(
                    func=ast.Name(id='SafeMock', ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg='name', value=ast.Constant(value=node.target.id)),
                        ast.keyword(arg='root', value=ast.Constant(value=True))
                    ]
                )
            )]
        else:
            except_body = [ast.Pass()]
        try_except_node = ast.Try(
            body=try_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=except_body
            )],
            orelse=[], finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

def collect_defined_names(tree):
    """Collects all names defined in the script."""
    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
    return defined_names

def transform_ast(script, target_functions):
    """
    Parses and transforms the AST to intercept function/method calls and wrap
    vulnerable constructs in try/except blocks.
    """
    # Replace tabs with spaces to help fix indentation issues.
    script = script.replace('\t', '    ')
    try:
        tree = ast.parse(dedent(script))
    except Exception as e:
        print(f"Error parsing script: {e}")
        tree = ast.parse("")
    defined_names = collect_defined_names(tree)
    transformer = ASTTransformer(target_functions, defined_names)
    try:
        transformed_tree = transformer.visit(tree)
    except Exception as e:
        print(f"Error during AST transformation: {e}")
        transformed_tree = tree
    ast.fix_missing_locations(transformed_tree)
    return transformed_tree

def extract_function_parameters(scripts, function_names):
    results = []
    for script in scripts:
        try:
            transformed_ast = transform_ast(script, function_names)
            namespace = create_mock_namespace()
            exec(compile(transformed_ast, filename="<ast>", mode="exec"), namespace)
            
            mock_registry = namespace.get('mock_registry', [])
            for func_name, mock_instance in mock_registry:
                if func_name in function_names:
                    for caller_chain, call_args, call_kwargs in mock_instance.call_args_list:
                        processed_args = [
                            repr(arg) if isinstance(arg, SafeMock) else arg
                            for arg in call_args
                        ]
                        processed_kwargs = {
                            k: repr(v) if isinstance(v, SafeMock) else v
                            for k, v in call_kwargs.items()
                        }
                        results.append({
                            'function': func_name,
                            'caller': ".".join(caller_chain) if caller_chain else "direct_call",
                            'args': processed_args,
                            'kwargs': processed_kwargs
                        })
        except Exception as e:
            print(f"Error processing script: {e}")
    return results

def remove_duplicates(input_list):
    def to_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, to_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(to_hashable(e) for e in obj)
        else:
            return obj

    seen = set()
    result = []
    for item in input_list:
        h = to_hashable(item)
        if h not in seen:
            seen.add(h)
            result.append(item)
    return result
