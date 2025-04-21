#ifndef BASIC_VM_DIALECT_H
#define BASIC_VM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace basicvm {

class BasicVMDialect : public Dialect {
public:
  explicit BasicVMDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "basicvm"; }
};

// Base class for all BasicVM operations
class BasicVMOp : public Op<BasicVMOp, OpTrait::OneResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "basicvm.operation"; }

  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &printer);
};


class HaltOp : public Op<HaltOp,
                OpTrait::ZeroResults,
                OpTrait::ZeroOperands,
                OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "basicvm.HaltOp"; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &printer);                                         \
    static void build(OpBuilder &builder, OperationState &result) {            \
    /* no operands, no results */                                            \
    }              
    static ArrayRef<StringRef> getAttributeNames() { return {}; }              \
    static ArrayRef<StringRef> getOperandNames()   { return {}; }              \
    static ArrayRef<StringRef> getResultNames()    { return {}; }     
};

// Operation class for stack operations
#define HANDLE_OP(OpClass)                                                      \
  class OpClass                                                                \
      : public Op<OpClass, OpTrait::ZeroResults, OpTrait::ZeroOperands> {      \
  public:                                                                      \
    using Op::Op;                                                              \
    static StringRef getOperationName() { return "basicvm." #OpClass; }       \
    static ParseResult parse(OpAsmParser &parser, OperationState &result);     \
    void print(OpAsmPrinter &printer);                                         \
    static void build(OpBuilder &builder, OperationState &result) {            \
      /* no operands, no results */                                            \
    }                                                                          \
    static ArrayRef<StringRef> getAttributeNames() { return {}; }              \
    static ArrayRef<StringRef> getOperandNames()   { return {}; }              \
    static ArrayRef<StringRef> getResultNames()    { return {}; }              \
  };

// Define operations for BasicVM opcodes
HANDLE_OP(PushOp);
HANDLE_OP(PopOp);
HANDLE_OP(AddOp);
HANDLE_OP(SubOp);
HANDLE_OP(MulOp);
HANDLE_OP(DivOp);
HANDLE_OP(PrintOp);
HANDLE_OP(StoreOp);
HANDLE_OP(LoadOp);
HANDLE_OP(JumpOp);
HANDLE_OP(JumpZeroOp);
HANDLE_OP(JumpNonZeroOp);
HANDLE_OP(CompareOp);
HANDLE_OP(AllocOp);
HANDLE_OP(FreeOp);
HANDLE_OP(StoreHeapOp);
HANDLE_OP(LoadHeapOp);

#undef HANDLE_OP

} // namespace basicvm
} // namespace mlir

#endif // BASIC_VM_DIALECT_H 