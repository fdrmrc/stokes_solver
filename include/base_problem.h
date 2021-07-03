#ifndef base_problem_include_file
#define base_problem_include_file

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/signals2.hpp>

#include <fstream>
#include <iostream>

#define FORCE_USE_OF_TRILINOS

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA


// Forward declare the tester class
template <typename Integral>
class BaseProblemTester;

using namespace dealii;

/**
 * Solve the Poisson problem, with Dirichlet or Neumann boundary conditions, on
 * all geometries that can be generated by the functions in the GridGenerator
 * namespace.
 */
template <int dim>
class BaseProblem : public ParameterAcceptor
{
public:
  /**
   * Constructor. Store component names and component masks.
   */
  BaseProblem(const unsigned int &n_components = 1,
              const std::string & problem_name = "");

  /**
   * Virtual destructor.
   */
  virtual ~BaseProblem() = default;

  /**
   * Output trivial info, like number of dofs, cells, threads, etc.
   */
  void
  print_system_info();

  /**
   * Main entry point of the problem.
   */
  void
  run();

  /**
   * Initialize the internal parameters with the given file.
   *
   * @param filename Name of the parameter file.
   */
  void
  initialize(const std::string &filename);

  /**
   * Parse a string as if it was a parameter file, and set the parameters
   * accordingly. Used mostly in the testers.
   *
   * @param par The string containing some parameters.
   */
  void
  parse_string(const std::string &par);

  /**
   * Default CopyData object, used in the WorkStream class.
   */
  using CopyData = MeshWorker::CopyData<2,1,1>;

  /**
   * Default ScratchData object, used in the workstream class.
   */
  using ScratchData = MeshWorker::ScratchData<dim>;

protected:
  /**
   * Assemble the local system matrix on `cell`, using `scratch` for FEValues
   * and other expensive scratch objects, and store the result in the `copy`
   * object. See the documentation of WorkStream for an explanation of how to
   * use this function.
   *
   * @param cell Cell on which we assemble the local matrix and rhs.
   * @param scratch Scratch object.
   * @param copy Copy object.
   */
  virtual void
  assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    CopyData &                                            copy);

  /**
   * Distribute the data that has been assembled by assemble_system_on_cell() to
   * the global matrix and rhs.
   *
   * @param copy The local data to distribute on the system matrix and rhs.
   */
  virtual void
  copy_one_cell(const CopyData &copy);

  /**
   * Generate the initial grid specified in the parameter file.
   */
  void
  make_grid();

  /**
   * Solve the global system.
   */
  virtual void
  solve();

  /**
   * Perform a posteriori error estimation, and store the results in the
   * `error_per_cell` vector.
   */
  virtual void
  estimate();

  /**
   * According to the chosen strategy, mark some cells for refinement.
   */
  void
  mark();

  /**
   * Refine the grid.
   */
  void
  refine_grid();

  /**
   * Initial setup: distribute degrees of freedom, make all vectors and matrices
   * of the right size, initialize functions and pointers.
   */
  virtual void
  setup_system();

  /**
   * A signal that is called at the end of setup_system()
   */
  boost::signals2::signal<void()> setup_system_call_back;

  /**
   * Actually loop over cells, and assemble the global system.
   */
  virtual void
  assemble_system();

  /**
   * Output the solution and the grid in a format that can be read by Paraview
   * or Visit.
   *
   * @param cycle A number identifying the refinement cycle.
   */
  virtual void
  output_results(const unsigned cycle) const;

  /**
   * Connect to this signal to add data vectors.
   */
  boost::signals2::signal<void(DataOut<dim> &)> add_data_vector;

  /**
   * Number of components.
   */
  const unsigned int n_components;

  /**
   * Global mpi communicator.
   */
  MPI_Comm mpi_communicator;

  /**
   * Number of threads to use for multi-threaded assembly.
   */
  int number_of_threads = -1;

  /**
   * Output only on processor zero.
   */
  ConditionalOStream pcout;

  /**
   * Timing information.
   */
  mutable TimerOutput timer;

  /**
   * The problem triangulation.
   */
  parallel::distributed::Triangulation<dim> triangulation;

  /**
   * The Finite Element space.
   *
   * This is a unique pointer to allow creation via parameter files.
   */
  std::unique_ptr<FiniteElement<dim>> fe;

  /**
   * The Mapping between reference and real elements.
   *
   * This is a unique pointer to allow creation via parameter files.
   */
  std::unique_ptr<MappingQGeneric<dim>> mapping;

  /**
   * Handler of degrees of freedom.
   */
  DoFHandler<dim> dof_handler;

  /**
   * Hanging nodes and essential boundary conditions.
   */
  AffineConstraints<double> constraints;

  /**
   * All degrees of freedom owned by this MPI process.
   */
  IndexSet locally_owned_dofs;

  /**
   * All degrees of freedom needed for output and error estimation.
   */
  IndexSet locally_relevant_dofs;

  /**
   * System matrix.
   */
  LA::MPI::SparseMatrix system_matrix;

  /**
   * A read only copy of the solution vector used for output and error
   * estimation.
   */
  LA::MPI::Vector locally_relevant_solution;

  /**
   * Solution vector.
   */
  LA::MPI::Vector solution;

  /**
   * The system right hand side. Read-write vector, containing only locally
   * owned dofs.
   */
  LA::MPI::Vector system_rhs;

  /**
   * Storage for local error estimator. This vector contains also values
   * associated to  artificial cells (i.e., it is of length
   * `triangulation.n_active_cells()`), but it is non-zero only on locally owned
   * cells. The estimate() method only fills locally owned cells.
   */
  Vector<float> error_per_cell;

  /**
   * Choose between `exact` estimator, `kelly` estimator, and `residual`
   * estimator.
   */
  std::string estimator_type = "kelly";

  /**
   * Choose between `global`, `fixed_fraction` (also known as Dorfler marking
   * strategy), and `fixed_number`.
   */
  std::string marking_strategy = "fixed_number";

  /**
   * Coarsening and refinement fractions.
   */
  std::pair<double, double> coarsening_and_refinement_factors = {0.03, 0.3};

  /**
   * Any cell with diameter smaller than this function evaluated at the center
   * of the cell, will be refined. This is performed `n_refimement` times before
   * starting the simulation.
   */
  FunctionParser<dim> pre_refinement;

  /**
   * Forcing term to put on the right hand side of the equation.
   */
  FunctionParser<dim> forcing_term;

  /**
   * Function used to compute errors.
   */
  FunctionParser<dim> exact_solution;

  /**
   * Non-homogeneous essential boundary conditions.
   */
  FunctionParser<dim> dirichlet_boundary_condition;

  /**
   * Non-homogeneous natural boundary conditions.
   */
  FunctionParser<dim> neumann_boundary_condition;

  /**
   * String used to generate the finite element space. Should be compatible with
   * FETools::get_fe_by_name().
   */
  std::string fe_name = "FESystem[FE_Q(2)^d - FE_Q(1)]";

  /**
   * Degree of the Mapping between reference and real elements.
   */
  unsigned int mapping_degree = 1;

  /**
   * Number of prerefiments to perform before the simulation starts.
   */
  unsigned int n_refinements = 4;

  /**
   * Number of solve-estimate-mark-refine cycles to perform.
   */
  unsigned int n_refinement_cycles = 7;

  /**
   * Output name for solution.
   */
  std::string output_filename = "stokes";

  /**
   * On which boundary ids we impose essential boundary conditions.
   */
  std::set<types::boundary_id> dirichlet_ids = {0,1,2,3};

  /**
   * On which boundary ids we impose natural boundary conditions.
   */
  std::set<types::boundary_id> neumann_ids = {};

  /**
   * Constants to use in the definitions of the various Function objects.
   */
  std::map<std::string, double> constants;

  /**
   * Expression to be used for the forcing term.
   */

  std::string forcing_term_expression = "0;0; 0";

  /**
   * Expression to be used for the exact solution.
   */
  std::string exact_solution_expression = "0;0;0";

  /**
   * Expression to be used for the Dirichlet boundary conditions.
   */
  std::string dirichlet_boundary_conditions_expression = "1;1;0";

  /**
   * Expression to be used for the Neumann boundary conditions.
   */
  std::string neumann_boundary_conditions_expression = "0;0;0";

  /**
   * Expression to be used for the pre-refinement function.
   */
  std::string pre_refinement_expression = "0";

  /**
   * Name of the GridGenerator function to call.
   */
  std::string grid_generator_function = "hyper_cube";

  /**
   * Arguments to pass to the GridGenerator function.
   */
  std::string grid_generator_arguments = "0: 1: true"; //colorize = true

  /**
   * A table used to output convergence errors.
   */
  ParsedConvergenceTable error_table;

  /**
   * Class used to store solver parameters, like maximum number of iterations,
   * absolute tolerance and relative tolerance.
   */
  ParameterAcceptorProxy<ReductionControl> solver_control;

  /**
   * Name of the tester class.
   */
  template <typename Integral>
  friend class BaseProblemTester;
};


template <typename ProblemType>
int
run(int argc, char **argv)
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
      std::string                      par_name = "";
      if (argc > 1)
        par_name = argv[1];

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(2);
      else
        deallog.depth_console(0);

      ProblemType base_problem;
      base_problem.initialize(par_name);
      base_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

#endif
