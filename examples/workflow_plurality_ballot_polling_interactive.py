# This is a starter script for interactively exploring the plurality ballot
# polling workflow example.

from examples.workflow_plurality_ballot_polling import create_workflow_data

# Run the workflow setup to get the data structures for inspection.
audit, contests, cvrs, sample = create_workflow_data()

# Extract the assertions for the contest, for easy access.
assertions = contests["mayor"].assertions

# Inspect the setup, for example:
#   contests["mayor"].tally
#   assertions.assertions["Alice v Bob"].margin
#   cvrs[:3], sample[:3]

# To calculate evidence and print the audit status:
#   assertions.set_p_values(sample)
#   audit.summarize_status(contests)
