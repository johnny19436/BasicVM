# Compiler and flags
CXX       = g++
CXXFLAGS  = -std=c++17 -Wall -Wextra -pedantic \
             -Wno-unused-parameter -Wno-unused-function

# LLVM/MLIR configuration
LLVM_CONFIG    = /opt/homebrew/opt/llvm/bin/llvm-config
LLVM_CXXFLAGS  = $(shell $(LLVM_CONFIG) --cxxflags)
LLVM_LDFLAGS   = $(shell $(LLVM_CONFIG) --ldflags)
LLVM_LIBS      = $(shell $(LLVM_CONFIG) --libs core support)

# which include + lib dirs
CXXFLAGS      += $(LLVM_CXXFLAGS) -I/opt/homebrew/opt/llvm/include
LDFLAGS        = $(LLVM_LDFLAGS) -L/opt/homebrew/opt/llvm/lib

# MLIR dynamic libraries (just -lXYZ, so the .dylib can satisfy the symbols)
MLIR_LIBS = \
	-lMLIRSupport \
	-lMLIRAnalysis \
	-lMLIRSideEffectInterfaces \
	-lMLIRCallInterfaces \
	-lMLIRControlFlowInterfaces \
	-lMLIRDataLayoutInterfaces \
	-lMLIRInferTypeOpInterface \
	-lMLIRIR \
	-lMLIRParser \
	-lMLIRAsmParser \
	-lMLIRBytecodeReader \
	-lMLIRPass \
	-lMLIRTransforms \
	-lMLIRTransformUtils \
	-lMLIRRewrite \
	-lMLIRArithDialect \
	-lMLIRFuncDialect \
	-lMLIRFunctionInterfaces

# PDL static archives (we still need to force‑load these to get every symbol)
PDL_STATIC = \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLDialect.a \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLLAST.a \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLLODS.a \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLLCodeGen.a \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLToPDLInterp.a \
	/opt/homebrew/opt/llvm/lib/libMLIRPDLInterpDialect.a \
	/opt/homebrew/opt/llvm/lib/libMLIRRewrite.a \
	/opt/homebrew/opt/llvm/lib/libMLIRRewritePDL.a \
	/opt/homebrew/opt/llvm/lib/libMLIRControlFlowDialect.a \
	/opt/homebrew/opt/llvm/lib/libMLIRMemRefDialect.a \
	/opt/homebrew/opt/llvm/lib/libMLIRInferIntRangeInterface.a \
	/opt/homebrew/opt/llvm/lib/libMLIRUBDialect.a

PARSER_STATIC = \
    /opt/homebrew/opt/llvm/lib/libMLIRParser.a \
    /opt/homebrew/opt/llvm/lib/libMLIRAsmParser.a \
    /opt/homebrew/opt/llvm/lib/libMLIRBytecodeReader.a


# Helper to emit a -force_load flag
define FORCE_LOAD
  -Wl,-force_load,$(1)
endef

# Targets
TARGETS       = vm_exec vm_basic
VM_SRCS       = main.cpp vm.cpp assembler.cpp mlir_integration.cpp mlir_compiler.cpp basic_vm_dialect.cpp
VM_OBJS       = $(VM_SRCS:.cpp=.o)
VM_BASIC_SRCS = vm_basic.cpp vm.cpp assembler.cpp
VM_BASIC_OBJS = $(VM_BASIC_SRCS:.cpp=.o)

all: $(TARGETS)

vm_exec: $(VM_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ \
	$(LLVM_LIBS) \
	-lMLIR \
	$(MLIR_LIBS)
	  

vm_basic: $(VM_BASIC_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(VM_OBJS) $(VM_BASIC_OBJS) $(TARGETS)

run: vm_exec
	./vm_exec

run_basic: vm_basic
	./vm_basic

.PHONY: all clean run run_basic
