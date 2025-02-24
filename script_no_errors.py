import ast
import inspect
from textwrap import dedent
from unittest.mock import MagicMock

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
        if len(self._chain) > 10:  # arbitrary limit to avoid runaway chains
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
            result = self
            self.index += 1
            return result
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
        return self
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
    Inherits operator implementations from SafeMock.
    """
    def __init__(self):
        super().__init__()
        self.terminal = True  # Always terminal

    def __call__(self, *args, **kwargs):
        return self

    def count(self):
        return 1

# --- A helper to safely evaluate expressions ---
def safe_eval(func):
    try:
        return func()
    except Exception:
        return SafeMock()

def create_mock_namespace():
    """Creates execution namespace with mock tracking."""
    mock_registry = []
    
    def create_and_register_mock(func_name):
        def wrapper(*args, **kwargs):
            remove_self = kwargs.pop("__remove_self", False)
            if remove_self and args:
                caller = args[0]
                args = args[1:]  # Remove the instance parameter
                caller_name = caller.name if isinstance(caller, SafeMock) else "unknown"
            else:
                caller_name = "direct_call"
            new_mock = SafeMock(name=func_name)
            new_mock.caller_name = caller_name
            new_mock(*args, **kwargs)
            mock_registry.append((func_name, new_mock))
            return new_mock
        return wrapper

    builtins_dict = (vars(__builtins__).copy() 
                     if inspect.ismodule(__builtins__) 
                     else __builtins__.copy())
    
    return {
        'mock_registry': mock_registry,
        'create_and_register_mock': create_and_register_mock,
        'MagicMock': MagicMock,
        'SafeMock': SafeMock,
        'TerminalSafeMock': TerminalSafeMock,
        'safe_eval': safe_eval,
        '__builtins__': builtins_dict
    }

class ASTTransformer(ast.NodeTransformer):
    """Transforms AST to mock targets, imports, and tracks mocks."""
    def __init__(self, target_functions, defined_names):
        self.target_functions = target_functions
        self.defined_names = defined_names

    def wrap_with_safe_eval(self, expr):
        # Wraps an expression in a lambda passed to safe_eval
        lambda_expr = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=expr
        )
        return ast.Call(
            func=ast.Name(id='safe_eval', ctx=ast.Load()),
            args=[lambda_expr],
            keywords=[]
        )

    def visit_Module(self, node):
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
            orelse=[],
            finalbody=[]
        )
        false_try = ast.Try(
            body=false_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=[ast.Pass()]
            )],
            orelse=[],
            finalbody=[]
        )
        node.body = [true_try]
        node.orelse = [false_try]
        node._wrapped_if = True
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        # Handle method calls on attributes (e.g. self.writer.saveAsTable(...))
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.target_functions:
            new_func = ast.Call(
                func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                args=[ast.Constant(value=node.func.attr)],
                keywords=[]
            )
            # Wrap the instance and each argument with safe_eval to catch errors during evaluation.
            new_instance = self.wrap_with_safe_eval(node.func.value)
            new_args = [new_instance] + [self.wrap_with_safe_eval(arg) for arg in node.args]
            new_keywords = [ast.keyword(arg=kw.arg, value=self.wrap_with_safe_eval(kw.value))
                            for kw in node.keywords]
            new_keywords.append(ast.keyword(arg="__remove_self", value=ast.Constant(value=True)))
            return ast.copy_location(ast.Call(
                func=new_func,
                args=new_args,
                keywords=new_keywords
            ), node)
        # Handle direct function calls (e.g. load(...))
        if isinstance(node.func, ast.Name) and node.func.id in self.target_functions:
            new_args = [self.wrap_with_safe_eval(arg) for arg in node.args]
            new_keywords = [ast.keyword(arg=kw.arg, value=self.wrap_with_safe_eval(kw.value))
                            for kw in node.keywords]
            return ast.copy_location(ast.Call(
                func=ast.Call(
                    func=ast.Name(id='create_and_register_mock', ctx=ast.Load()),
                    args=[ast.Constant(value=node.func.id)],
                    keywords=[]
                ),
                args=new_args,
                keywords=new_keywords
            ), node)
        return node

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
        return new_nodes if len(new_nodes) > 1 else new_nodes[0] if new_nodes else None

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
            orelse=[],
            finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        try_body = [node]
        target = node.target
        if isinstance(target, ast.Name):
            except_body = [
                ast.Assign(
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
            ]
        else:
            except_body = [ast.Pass()]
        try_except_node = ast.Try(
            body=try_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=except_body
            )],
            orelse=[],
            finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        try_body = [node]
        if isinstance(node.target, ast.Name):
            except_body = [
                ast.Assign(
                    targets=[ast.copy_location(ast.Name(id=node.target.id, ctx=ast.Store()), node.target)],
                    value=ast.Call(
                        func=ast.Name(id='SafeMock', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg='name', value=ast.Constant(value=node.target.id)),
                            ast.keyword(arg='root', value=ast.Constant(value=True))
                        ]
                    )
                )
            ]
        else:
            except_body = [ast.Pass()]
        try_except_node = ast.Try(
            body=try_body,
            handlers=[ast.ExceptHandler(
                type=ast.Name(id='Exception', ctx=ast.Load()),
                name=None,
                body=except_body
            )],
            orelse=[],
            finalbody=[]
        )
        return ast.copy_location(try_except_node, node)

def collect_defined_names(tree):
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

script = '''import sys
import json
import logging
import math
from datetime import datetime
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import numpy as np

# ----- Logging Setup -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {datetime.now() - start_time}")
        return result
    return wrapper

# ----- Base Classes -----
class DataProcessor:
    def __init__(self, spark):
        self.spark = spark
        self.metrics = {}
    
    def update_metric(self, key, value):
        self.metrics[key] = value

    def get_metrics(self):
        return self.metrics

    def process(self):
        raise NotImplementedError("Subclasses must implement process method")

class AdvancedDataWriter(DataProcessor):
    def __init__(self, spark):
        super().__init__(spark)
        self.write_count = 0

    @log_execution_time
    def write_to_glue(self, df, database, table, path, mode="overwrite"):
        df.write.format("parquet").mode(mode).option("path", path).saveAsTable(f"{database}.{a}")
        self.write_count += 1
        self.update_metric("records_written", df.count())
        self.update_metric("write_operations", self.write_count)

# ----- Multi-Table ETL Process -----
class MultiTableETL(DataProcessor):
    def __init__(self, spark, config):
        super().__init__(spark)
        self.config = config
        self.writer = AdvancedDataWriter(spark)
        self.error_logs = []

    sensor_schema = StructType([
        StructField("sensor_id", StringType(), True),
        StructField("reading", DoubleType(), True),
        StructField("timestamp", IntegerType(), True)
    ])

    metadata_schema = StructType([
        StructField("sensor_id", StringType(), True),
        StructField("location", StringType(), True),
        StructField("device_type", StringType(), True)
    ])

    lookup_schema = StructType([
        StructField("device_type", StringType(), True),
        StructField("calibration_factor", DoubleType(), True)
    ])
    
    @log_execution_time
    def read_sensor_data(self):
        try:
            df = self.spark.read.schema(self.sensor_schema).format("parquet").load(self.config["sensor_data_path"])
            self.update_metric("sensor_records", df.count())
            return df
        except Exception as e:
            self.error_logs.append(f"Error reading sensor data: {str(e)}")
            raise

    @log_execution_time
    def read_metadata(self):
        try:
            df = self.spark.read.schema(self.metadata_schema).format("parquet").load(self.config["metadata_path"])
            self.update_metric("metadata_records", df.count())
            return df
        except Exception as e:
            self.error_logs.append(f"Error reading metadata: {str(e)}")
            raise

    @log_execution_time
    def read_lookup(self):
        try:
            df = self.spark.read.schema(self.lookup_schema).format("parquet").load(self.config["lookup_path"])
            self.update_metric("lookup_records", df.count())
            return df
        except Exception as e:
            self.error_logs.append(f"Error reading lookup: {str(e)}")
            raise

    @staticmethod
    def compute_transformation(x):
        try:
            return float(math.log(abs(x) + 1) * np.sin(x))
        except Exception:
            return None

    compute_transformation_udf = F.udf(compute_transformation, DoubleType())

    @log_execution_time
    def transform_sensor_data(self, sensor_df):
        transformed_df = sensor_df.withColumn("transformed_reading", self.compute_transformation_udf(F.col("reading")))
        filtered_df = transformed_df.filter(F.col("transformed_reading").isNotNull())
        self.update_metric("valid_sensor_records", filtered_df.count())
        return filtered_df

    @log_execution_time
    def join_datasets(self, sensor_df, metadata_df, lookup_df):
        joined_df = sensor_df.join(metadata_df, on="sensor_id", how="left")
        final_df = joined_df.join(lookup_df, on="device_type", how="left")
        final_df = final_df.withColumn("calibrated_reading", F.col("transformed_reading") * F.col("calibration_factor"))
        self.update_metric("joined_records", final_df.count())
        return final_df

    @log_execution_time
    def window_analysis(self, df):
        window_spec = Window.partitionBy("location").orderBy("timestamp")
        df_with_rank = df.withColumn("reading_rank", F.rank().over(window_spec))
        agg_df = df_with_rank.groupBy("location").agg(
            F.avg("calibrated_reading").alias("avg_calibrated"),
            F.max("calibrated_reading").alias("max_calibrated")
        )
        self.update_metric("locations_analyzed", agg_df.count())
        return df_with_rank, agg_df

    @log_execution_time
    def enrich_with_time_features(self, df):
        current_date = int(str(datetime.now().strftime("%Y-%m-%d")))
        enriched_df = df.withColumn("processing_time", F.lit(datetime.now().isoformat()))
        enriched_df = enriched_df.withColumn("processing_date", F.lit(current_date))
        return enriched_df

    @log_execution_time
    def process(self):
        try:
            sensor_df = self.read_sensor_data()
            metadata_df = self.read_metadata()
            lookup_df = self.read_lookup()
            transformed_sensor_df = self.transform_sensor_data(sensor_df)
            joined_df = self.join_datasets(transformed_sensor_df, metadata_df, lookup_df)
            df_with_rank, agg_df = self.window_analysis(joined_df)
            enriched_df = self.enrich_with_time_features(df_with_rank)
            self.writer.write_to_glue(
                enriched_df,
                self.config["database"],
                self.config["detailed_table"],
                self.config["detailed_output_path"]
            )
            self.writer.write_to_glue(
                agg_df,
                self.config["database"],
                self.config["aggregated_table"],
                self.config["aggregated_output_path"]
            )
            self.update_metric("final_records", enriched_df.count())
            return enriched_df, agg_df
        except Exception as e:
            logger.error(f"ETL process failed: {str(e)}")
            self.error_logs.append(f"ETL process failed: {str(e)}")
            raise

if __name__ == "__main__":
    config = {
        "sensor_data_path": "s3a://my-bucket/sensor_data/",
        "metadata_path": "s3a://my-bucket/metadata/",
        "lookup_path": "s3a://my-bucket/lookup/",
        "detailed_output_path": "s3a://my-bucket/processed/detailed/",
        "aggregated_output_path": "s3a://my-bucket/processed/aggregated/",
        "database": "iot_db",
        "detailed_table": "sensor_detailed",
        "aggregated_table": "sensor_aggregated"
    }

    spark = SparkSession.builder \
        .appName("ComplexMultiTableETL") \
        .config("spark.sql.shuffle.partitions", "20") \
        .enableHiveSupport() \
        .getOrCreate()

    etl_job = MultiTableETL(spark, config)
    detailed_df, aggregated_df = etl_job.process()

    metrics = etl_job.get_metrics()
    logger.info("ETL Job Metrics:\\n" + json.dumps(metrics, indent=2))
    if etl_job.error_logs:
        logger.error("Errors encountered:\\n" + json.dumps(etl_job.error_logs, indent=2))

    spark.stop()
elif 20>10:
    config = {
        "sensor_data_path": "s3a://my-bucket/sensor_data/",
        "metadata_path": "s3a://my-bucket/metadata/",
        "lookup_path": "s3a://my-bucket/lookup/",
        "detailed_output_path": "s3a://my-bucket/processed/detailed/",
        "aggregated_output_path": "s3a://my-bucket/processed/aggregated/",
        "database": "iot_db",
        "detailed_table": "sensor_detailed",
        "aggregated_table": "sensor_aggregated"
    }

    spark = SparkSession.builder \
        .appName("ComplexMultiTableETL") \
        .config("spark.sql.shuffle.partitions", "20") \
        .enableHiveSupport() \
        .getOrCreate()

    etl_job = MultiTableETL(spark, config)
    detailed_df, aggregated_df = etl_job.process()

    metrics = etl_job.get_metrics()
    logger.info("ETL Job Metrics:\\n" + json.dumps(metrics, indent=2))
    if etl_job.error_logs:
        logger.error("Errors encountered:\\n" + json.dumps(etl_job.error_logs, indent=2))

    spark.stop()
else:
    config = {
        "sensor_data_path": "s3a://my-bucket/sensor_data/",
        "metadata_path": "s3a://my-bucket/metadata/",
        "lookup_path": "s3a://my-bucket/lookup/",
        "detailed_output_path": "s3a://my-bucket/processed/detailed/",
        "aggregated_output_path": "s3a://my-bucket/processed/aggregated/",
        "database": "iot_db",
        "detailed_table": "sensor_detailed",
        "aggregated_table": "sensor_aggregated"
    }

    spark = SparkSession.builder \
        .appName("ComplexMultiTableETL") \
        .config("spark.sql.shuffle.partitions", "20") \
        .enableHiveSupport() \
        .getOrCreate()

    etl_job = MultiTableETL(spark, config)
    detailed_df, aggregated_df = etl_job.process()

    metrics = etl_job.get_metrics()
    logger.info("ETL Job Metrics:\\n" + json.dumps(metrics, indent=2))
    if etl_job.error_logs:
        logger.error("Errors encountered:\\n" + json.dumps(etl_job.error_logs, indent=2))

    spark.stop()
'''

scripts = [script]
parameter = remove_duplicates(extract_function_parameters(scripts, ['saveAsTable']))
parameter
