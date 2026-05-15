# tools/calculator.py

from loguru import logger
import ast
import operator


# Safe allowed operators — no exec, no eval of arbitrary code
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,     # unary minus e.g. -5
    ast.UAdd: operator.pos,     # unary plus  e.g. +5
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": lambda x: x ** 0.5,
    "pow": pow,
}


class CalculatorTool:
    """
    Safe math expression evaluator.

    Uses Python's AST parser to evaluate expressions
    without allowing arbitrary code execution.

    Prevents: exec, eval, imports, function calls outside whitelist,
              attribute access, subscripts, and all other dangerous ops.
    """

    name = "calculator"
    description = (
        "Evaluates mathematical expressions. "
        "Use this for any arithmetic, algebra, or numeric computation. "
        "Input must be a valid math expression string. "
        "Examples: '2 + 2', '(10 * 5) / 2', 'sqrt(144)', '2 ** 10'"
    )

    def _safe_eval(self, node: ast.AST) -> float:
        """
        Recursively evaluates an AST node.
        Only allows whitelisted operations and functions.
        """
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            return ALLOWED_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
            operand = self._safe_eval(node.operand)
            return ALLOWED_OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")
            args = [self._safe_eval(arg) for arg in node.args]
            return ALLOWED_FUNCTIONS[func_name](*args)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def run(self, expression: str) -> str:
        """
        Evaluates a math expression string and returns the result.
        Returns an error message string if evaluation fails —
        so the agent can read the error and decide what to do.
        """
        logger.info(f"Calculator input: '{expression}'")

        try:
            # Clean up the expression
            expression = expression.strip()

            # Parse into AST — never executes code
            tree = ast.parse(expression, mode="eval")

            result = self._safe_eval(tree)

            # Round floats to avoid 0.1+0.2 = 0.30000000000004 style noise
            if isinstance(result, float):
                result = round(result, 10)
                # Show as int if it's a whole number
                if result == int(result):
                    result = int(result)

            logger.info(f"Calculator result: {expression} = {result}")
            return str(result)

        except ZeroDivisionError:
            logger.warning(f"Division by zero in: '{expression}'")
            return "Error: division by zero"

        except ValueError as e:
            logger.warning(f"Calculator ValueError: {e}")
            return f"Error: {e}"

        except SyntaxError:
            logger.warning(f"Invalid expression syntax: '{expression}'")
            return f"Error: invalid expression '{expression}'"

        except Exception as e:
            logger.error(f"Calculator unexpected error: {e}")
            return f"Error: {e}"