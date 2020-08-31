{ pkgs, stdenv, fetchFromGitHub, python, fetchPypi, matplotlib, scipy, numpy }:

let
  pyPackages = py: with py; [

  ];

in
  python.pkgs.buildPythonPackage rec {
    pname = "matplotlib-venn";
    version = "0.11.5";
    propagatedBuildInputs = with pkgs; [ matplotlib scipy numpy ];
    doCheck = false;

    src = pkgs.fetchFromGitHub {
      owner = "konstantint";
      repo = "matplotlib-venn";
      rev = "c26796c9925bdac512edf48387452fbd1848c791";
      sha256 = "0ca2zpdgzpzxhfbmpd453dsdkgc9bj6kh4j75fd85ds2mbry1n0j";
    };
  }
