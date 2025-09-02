using ElectronSpinDynamics
using Documenter

DocMeta.setdocmeta!(
  ElectronSpinDynamics, :DocTestSetup, :(using ElectronSpinDynamics); recursive=true
)

makedocs(;
  modules=[ElectronSpinDynamics],
  authors="Kentaro Hino",
  sitename="ElectronSpinDynamics.jl",
  format=Documenter.HTML(; edit_link="main", assets=String[]),
  pages=["Home" => "index.md"],
)
