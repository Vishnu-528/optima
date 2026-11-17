"""Error reporting for the Optima compiler."""


class OptimaError(Exception):
    """A compile error tied to a position in the source file."""

    def __init__(self, message, line=None, col=None):
        super().__init__(message)
        self.message = message
        self.line = line  # 1-based
        self.col = col    # 1-based

    def format(self, source=None, filename="<input>"):
        """Render as 'file:line:col: error: msg' plus the offending line."""
        if self.line is None:
            return f"{filename}: error: {self.message}"
        out = f"{filename}:{self.line}:{self.col}: error: {self.message}"
        if source is not None:
            lines = source.splitlines()
            if 1 <= self.line <= len(lines):
                text = lines[self.line - 1].replace("\t", " ")
                out += f"\n    {text}\n    {' ' * (self.col - 1)}^"
        return out
