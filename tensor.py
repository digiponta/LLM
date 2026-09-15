# tensor.py
#
# Tensor abstraction for the homemade virtual GPU.
#
# Dependencies:
#   gpu_memory.py
#   gpu_isa.py
#   gpu_compute.py
#
# NumPy is intentionally not used.

from typing import List, Tuple, Union, Optional

from gpu_memory import GPUMemory
from gpu_isa import Opcode, Instruction, GPUProgram, reg, mem, imm
from gpu_compute import GPUComputeCore


Number = Union[int, float]


def _infer_shape(data) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if len(data) == 0:
        return (0,)
    first_shape = _infer_shape(data[0])
    for item in data:
        if _infer_shape(item) != first_shape:
            raise ValueError("Irregular nested list cannot be converted to Tensor.")
    return (len(data),) + first_shape


def _flatten(data) -> List[float]:
    if not isinstance(data, list):
        return [float(data)]
    result = []
    for item in data:
        result.extend(_flatten(item))
    return result


def _numel(shape: Tuple[int, ...]) -> int:
    if len(shape) == 0:
        return 1
    n = 1
    for dimension in shape:
        n *= dimension
    return n


def _unflatten(values: List[float], shape: Tuple[int, ...]):
    if len(shape) == 0:
        return values[0]
    if len(shape) == 1:
        return values[:shape[0]]
    stride = _numel(shape[1:])
    result = []
    for i in range(shape[0]):
        start = i * stride
        end = start + stride
        result.append(_unflatten(values[start:end], shape[1:]))
    return result


class TensorRuntime:
    def __init__(self, memory_size: int = 1_000_000, num_registers: int = 32):
        self.memory = GPUMemory(size=memory_size)
        self.core = GPUComputeCore(memory=self.memory, num_registers=num_registers)

    def run(self, program: GPUProgram) -> None:
        self.core.run(program)


_default_runtime: Optional[TensorRuntime] = None


def get_default_runtime() -> TensorRuntime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = TensorRuntime()
    return _default_runtime


def reset_default_runtime() -> None:
    global _default_runtime
    _default_runtime = TensorRuntime()


class Tensor:
    def __init__(
        self,
        data=None,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        runtime: Optional[TensorRuntime] = None,
        address: Optional[int] = None,
    ):
        if runtime is None:
            runtime = get_default_runtime()
        self.runtime = runtime

        if address is not None:
            if shape is None:
                raise ValueError("shape is required when address is specified.")
            self.shape = tuple(shape)
            self.size = _numel(self.shape)
            self.address = address
            return

        if data is None:
            if shape is None:
                raise ValueError("data or shape is required.")
            self.shape = tuple(shape)
            self.size = _numel(self.shape)
            values = [0.0] * self.size
        else:
            self.shape = _infer_shape(data)
            values = _flatten(data)
            self.size = len(values)

        self.address = self.runtime.memory.allocate(self.size)
        self.runtime.memory.write(self.address, values)

    @classmethod
    def zeros(cls, shape, runtime=None):
        return cls(shape=tuple(shape), runtime=runtime)

    @classmethod
    def ones(cls, shape, runtime=None):
        tensor = cls.zeros(shape, runtime=runtime)
        tensor.runtime.memory.fill(tensor.address, tensor.size, 1.0)
        return tensor

    @classmethod
    def full(cls, shape, value, runtime=None):
        tensor = cls.zeros(shape, runtime=runtime)
        tensor.runtime.memory.fill(tensor.address, tensor.size, float(value))
        return tensor

    def flat(self) -> List[float]:
        return self.runtime.memory.read(self.address, self.size)

    def tolist(self):
        return _unflatten(self.flat(), self.shape)

    def item(self) -> float:
        if self.size != 1:
            raise ValueError("item() requires a tensor with exactly one element.")
        return self.runtime.memory.read_scalar(self.address)

    def __repr__(self):
        return f"Tensor({self.tolist()}, shape={self.shape}, gpu_address={self.address})"

    def _binary_op(self, other, opcode: Opcode):
        result = Tensor.zeros(self.shape, runtime=self.runtime)

        if isinstance(other, Tensor):
            if other.runtime is not self.runtime:
                raise ValueError("Tensor runtimes must match.")
            if self.shape != other.shape:
                raise ValueError(f"Tensor shapes must match: {self.shape} != {other.shape}")

            for i in range(self.size):
                program = GPUProgram()
                program.add(Instruction(opcode=Opcode.LOAD, dst=reg(0), src1=mem(self.address + i)))
                program.add(Instruction(opcode=Opcode.LOAD, dst=reg(1), src1=mem(other.address + i)))
                program.add(Instruction(opcode=opcode, dst=reg(2), src1=reg(0), src2=reg(1)))
                program.add(Instruction(opcode=Opcode.STORE, dst=mem(result.address + i), src1=reg(2)))
                program.add(Instruction(opcode=Opcode.HALT))
                self.runtime.run(program)
            return result

        if isinstance(other, (int, float)):
            value = float(other)
            for i in range(self.size):
                program = GPUProgram()
                program.add(Instruction(opcode=Opcode.LOAD, dst=reg(0), src1=mem(self.address + i)))
                program.add(Instruction(opcode=Opcode.MOV, dst=reg(1), src1=imm(value)))
                program.add(Instruction(opcode=opcode, dst=reg(2), src1=reg(0), src2=reg(1)))
                program.add(Instruction(opcode=Opcode.STORE, dst=mem(result.address + i), src1=reg(2)))
                program.add(Instruction(opcode=Opcode.HALT))
                self.runtime.run(program)
            return result

        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other):
        return self._binary_op(other, Opcode.ADD)

    def __sub__(self, other):
        return self._binary_op(other, Opcode.SUB)

    def __mul__(self, other):
        return self._binary_op(other, Opcode.MUL)

    def __truediv__(self, other):
        return self._binary_op(other, Opcode.DIV)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires Tensor.")
        if self.runtime is not other.runtime:
            raise ValueError("Tensor runtimes must match.")
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication currently supports 2-D tensors only.")

        m = self.shape[0]
        k = self.shape[1]
        k2 = other.shape[0]
        n = other.shape[1]

        if k != k2:
            raise ValueError(f"Matrix dimensions do not match: {self.shape} @ {other.shape}")

        result = Tensor.zeros((m, n), runtime=self.runtime)
        program = GPUProgram()
        program.add(
            Instruction(
                opcode=Opcode.MATMUL,
                dst=mem(result.address),
                src1=mem(self.address),
                src2=mem(other.address),
                extra={"m": m, "n": n, "k": k},
            )
        )
        program.add(Instruction(opcode=Opcode.HALT))
        self.runtime.run(program)
        return result

    def softmax(self):
        result = Tensor.zeros(self.shape, runtime=self.runtime)
        program = GPUProgram()
        program.add(
            Instruction(
                opcode=Opcode.SOFTMAX,
                dst=mem(result.address),
                src1=mem(self.address),
                size=self.size,
            )
        )
        program.add(Instruction(opcode=Opcode.HALT))
        self.runtime.run(program)
        return result

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(x) for x in shape)
        if _numel(shape) != self.size:
            raise ValueError("reshape changes number of elements.")
        return Tensor(shape=shape, runtime=self.runtime, address=self.address)

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("transpose() currently supports 2-D tensors only.")

        rows, cols = self.shape
        result = Tensor.zeros((cols, rows), runtime=self.runtime)

        for row in range(rows):
            for col in range(cols):
                src_index = row * cols + col
                dst_index = col * rows + row
                value = self.runtime.memory.read_scalar(self.address + src_index)
                self.runtime.memory.write_scalar(result.address + dst_index, value)

        return result

    @property
    def T(self):
        return self.transpose()

    def clone(self):
        result = Tensor.zeros(self.shape, runtime=self.runtime)
        self.runtime.memory.copy(result.address, self.address, self.size)
        return result


if __name__ == "__main__":
    print("================================")
    print("Tensor Test")
    print("================================")

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    print("A =", a)
    print("B =", b)
    print("A + B =", a + b)
    print("A * B =", a * b)
    print("A * 10 =", a * 10)

    s = a.softmax()
    print("softmax(A) =", s)
    print("softmax sum =", sum(s.flat()))

    matrix_a = Tensor([[1, 2], [3, 4]])
    matrix_b = Tensor([[5, 6], [7, 8]])
    matrix_c = matrix_a @ matrix_b

    print("A @ B =", matrix_c)
    print("A.T =", matrix_a.T)

    x = Tensor([1, 2, 3, 4, 5, 6])
    print("X.reshape(2,3) =", x.reshape(2, 3))

    a.runtime.memory.info()
