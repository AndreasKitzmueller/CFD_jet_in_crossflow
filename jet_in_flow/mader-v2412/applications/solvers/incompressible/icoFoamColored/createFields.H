Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

dimensionedScalar nu
(
    "nu",
    dimViscosity,
    transportProperties
);

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, mesh.solutionDict().subDict("PISO"), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

// --- Passive tracer fields for coloring two streams
volScalarField cA
(
    IOobject
    (
        "cA",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,   // read if file exists in time dir
        IOobject::AUTO_WRITE         // write every time step
    ),
    mesh,
    dimensionedScalar("zero", dimless, 0)
);

volScalarField cB
(
    IOobject
    (
        "cB",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("zero", dimless, 0)
);

// Molecular diffusivity for the tracers (m^2/s)
// You can also read this from transportProperties if you prefer.
// Start with 0 for sharpest interface; try 1e-8..1e-6 if you want smoothing.
dimensionedScalar D("D", dimensionSet(0,2,-1,0,0,0,0), 0);

