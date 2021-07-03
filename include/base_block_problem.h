#ifndef base_block_problem_include_file
#define base_block_problem_include_file

#include "base_problem.h"

// Forward declare the tester class
template <typename Integral>
class BaseBlockProblemTester;

using namespace dealii;

/**
 * Construct a BaseBlockProblem.
 */
template <int dim>
class BaseBlockProblem : public BaseProblem<dim>
{
public:
  /**
   * Constructor. Store component names and component masks.
   */
  BaseBlockProblem(const std::vector<std::string> component_names = {{"u"}},
                   const std::string &            problem_name    = "");

  /**
   * Virtual destructor.
   */
  virtual ~BaseBlockProblem() = default;

  /**
   * Default CopyData object, used in the WorkStream class.
   */
  using CopyData = typename BaseProblem<dim>::CopyData;

  /**
   * Default ScratchData object, used in the workstream class.
   */
  using ScratchData = typename BaseProblem<dim>::ScratchData;

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
    CopyData &                                            copy) override;

  /**
   * Distribute the data that has been assembled by assemble_system_on_cell() to
   * the global matrix and rhs.
   *
   * @param copy The local data to distribute on the system matrix and rhs.
   */
  virtual void
  copy_one_cell(const CopyData &copy) override;

  virtual void
  setup_system() override;

  virtual void
  assemble_system() override;

  virtual void
  solve() override;

  const std::vector<std::string> component_names;

  /**
   * Dofs per block
   */
  std::vector<types::global_dof_index> dofs_per_block;

  /**
   * All degrees of freedom owned by this MPI process.
   */
  std::vector<IndexSet> locally_owned_dofs;

  /**
   * All degrees of freedom needed for output and error estimation.
   */
  std::vector<IndexSet> locally_relevant_dofs;



  /**
   * System matrix.
   */
  LA::MPI::BlockSparseMatrix system_block_matrix;


  /*
   * Preconditioner matrix
   */
  LA::MPI::BlockSparseMatrix preconditioner_matrix;


  /**
   * A read only copy of the solution vector used for output and error
   * estimation.
   */
  LA::MPI::BlockVector locally_relevant_block_solution;

  /**
   * Solution vector.
   */
  LA::MPI::BlockVector block_solution;

  /**
   * The system right hand side. Read-write vector, containing only locally
   * owned dofs.
   */
  LA::MPI::BlockVector system_block_rhs;

  /**
   * Extractor to use in the assembly routine.
   */
  FEValuesExtractors::Vector velocity;

  /**
   * Extractor to use in the assembly routine.
   */
  FEValuesExtractors::Scalar pressure;
  /*
   * Name of the tester class.
   */
  template <typename Integral>
  friend class BaseBlockProblemTester;
};


#endif
