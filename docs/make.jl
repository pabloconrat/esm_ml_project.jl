using esm_ml_project
using Documenter

DocMeta.setdocmeta!(esm_ml_project, :DocTestSetup, :(using esm_ml_project); recursive=true)

makedocs(;
    modules=[esm_ml_project],
    authors="Pablo Conrat <pablo.conrat@tum.de>",
    sitename="esm_ml_project.jl",
    format=Documenter.HTML(;
        canonical="https://pabloconrat.github.io/esm_ml_project.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pabloconrat/esm_ml_project.jl",
    devbranch="main",
)
