-- Define the project
set_project("MyProject")

-- Set common configurations
set_languages("c++17")  -- Assuming C++17 is used, modify if needed
add_rules("mode.debug", "mode.release")

-- Target: inference_engine (static library)
target("inference_engine")
    set_kind("static")  -- Static library
    add_includedirs("inference_engine/include")  -- Add include directory for this library
    add_files("inference_engine/src/*.cpp")  -- Add source files from the src folder

-- Target: interactive_shell (binary)
target("interactive_shell")
    set_kind("binary")  -- Binary executable
    add_includedirs("interactive_shell/include", "inference_engine/include")  -- Add include directories
    add_files("interactive_shell/src/*.cpp")  -- Add source files from the src folder
    add_deps("inference_engine")  -- This binary depends on the inference_engine library

-- Target: tests (binary)
target("tests")
    set_kind("binary")  -- Binary executable
    add_includedirs("inference_engine/include")  -- Include the inference engine headers
    add_files("tests/src/*.cpp")  -- Add test source files
    add_deps("inference_engine")  -- This binary also depends on the inference_engine library

-- Target: integration (binary)
target("integration")
    set_kind("binary")  -- Binary executable
    add_includedirs("inference_engine/include")  -- Include the inference engine headers
    add_files("integration/src/*.cpp")  -- Add integration test source files
    add_deps("inference_engine")  -- This binary also depends on the inference_engine library
