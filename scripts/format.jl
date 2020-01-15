#!/usr/bin/env julia

using JuliaFormatter

format(
    joinpath(@__DIR__, ".."),
    overwrite = true,
    verbose = true,
    indent = 4,
    margin = 92,
    always_for_in = true,
    whitespace_typedefs = false,
    whitespace_ops_in_indices = false,
    remove_extra_newlines = true,
)
