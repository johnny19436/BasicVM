#include "basic_vm_dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::basicvm;

//===----------------------------------------------------------------------===//
// BasicVM Dialect
//===----------------------------------------------------------------------===//

BasicVMDialect::BasicVMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<BasicVMDialect>()) {
  
  // Register operations
  addOperations<
    #define GET_OP_LIST
    HaltOp,
    PushOp,
    PopOp,
    AddOp,
    SubOp,
    MulOp,
    DivOp,
    PrintOp,
    StoreOp,
    LoadOp,
    JumpOp,
    JumpZeroOp,
    JumpNonZeroOp,
    CompareOp,
    AllocOp,
    FreeOp,
    StoreHeapOp,
    LoadHeapOp
    #undef GET_OP_LIST
  >();
}

//===----------------------------------------------------------------------===//
// BasicVM Operations
//===----------------------------------------------------------------------===//

static LogicalResult parseBasicVMOp(OpAsmParser &parser, OperationState &result) {
  // Just the basic parsing for now
  return success();
}

static void printBasicVMOp(OpAsmPrinter &printer, Operation *op) {
  printer << op->getName();
}

// Define parse and print for the base op
ParseResult BasicVMOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBasicVMOp(parser, result);
}

void BasicVMOp::print(OpAsmPrinter &printer) {
  printBasicVMOp(printer, getOperation());
}

// Macro to define parse and print for all operations
#define DEFINE_OP_METHODS(OpClass)                                       \
  ParseResult OpClass::parse(OpAsmParser &parser, OperationState &result) { \
    return parseBasicVMOp(parser, result);                               \
  }                                                                      \
  void OpClass::print(OpAsmPrinter &printer) {                           \
    printBasicVMOp(printer, getOperation());                             \
  }

// Define parse and print methods for all ops
DEFINE_OP_METHODS(HaltOp)
DEFINE_OP_METHODS(PushOp)
DEFINE_OP_METHODS(PopOp)
DEFINE_OP_METHODS(AddOp)
DEFINE_OP_METHODS(SubOp)
DEFINE_OP_METHODS(MulOp)
DEFINE_OP_METHODS(DivOp)
DEFINE_OP_METHODS(PrintOp)
DEFINE_OP_METHODS(StoreOp)
DEFINE_OP_METHODS(LoadOp)
DEFINE_OP_METHODS(JumpOp)
DEFINE_OP_METHODS(JumpZeroOp)
DEFINE_OP_METHODS(JumpNonZeroOp)
DEFINE_OP_METHODS(CompareOp)
DEFINE_OP_METHODS(AllocOp)
DEFINE_OP_METHODS(FreeOp)
DEFINE_OP_METHODS(StoreHeapOp)
DEFINE_OP_METHODS(LoadHeapOp)

#undef DEFINE_OP_METHODS 
