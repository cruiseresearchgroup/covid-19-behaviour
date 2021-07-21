{ pkgs ? import <nixpkgs> {} }:
let
  my-python-packages = python-packages: with python-packages; [
    pandas
    notebook
    matplotlib
    scikitlearn
    seaborn
    python-language-server
    pylint
    jupyterlab
    (python-packages.callPackage ./nix/matplotlibvenn.nix {})
  ]; 

  R-with-my-packages = pkgs.rWrapper.override{ packages = with pkgs.rPackages; [ 
    tidyverse
    corrplot 
    rstan
    bayesplot
    lubridate
    reticulate
    proxy
  ]; };
  python-with-my-packages = pkgs.python3.withPackages my-python-packages;
  
in
pkgs.mkShell {
  name = "analysis";
  buildInputs = with pkgs; [
    python-with-my-packages
    R-with-my-packages
    evince
    miller
  ];
}
