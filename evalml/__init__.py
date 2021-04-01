import warnings
import woodwork as ww
from woodwork.logical_types import LogicalType


ww.type_system.reset_defaults()


class Integer(LogicalType):
    primary_dtype = 'int64'
    standard_tags = {'numeric'}


class Boolean(LogicalType):
    primary_dtype = 'bool'


class String(LogicalType):
    primary_dtype = 'object'


int_inference_fn = ww.type_system.inference_functions[ww.logical_types.Integer]
bool_inference_fn = ww.type_system.inference_functions[ww.logical_types.Boolean]
str_inference_fn = ww.type_system.inference_functions[ww.logical_types.NaturalLanguage]


ww.type_system.remove_type(ww.logical_types.Integer)
ww.type_system.remove_type(ww.logical_types.Boolean)
ww.type_system.add_type(Integer, int_inference_fn)
ww.type_system.add_type(Boolean, bool_inference_fn)
ww.type_system.add_type(String, str_inference_fn)
ww.type_system.default_type = String
ww.type_system.remove_type(ww.logical_types.NaturalLanguage)

# hack to prevent warnings from skopt
# must import sklearn first
import sklearn
import evalml.demos
import evalml.model_family
import evalml.objectives
import evalml.pipelines
import evalml.preprocessing
import evalml.problem_types
import evalml.utils
import evalml.data_checks
from evalml.automl import AutoMLSearch
from evalml.utils import print_info
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import skopt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'The following selectors were not present in your DataTable')


__version__ = '0.21.0'
