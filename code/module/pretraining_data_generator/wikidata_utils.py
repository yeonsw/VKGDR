#Code reference: https://github.com/jamesmishra/mysqldump-to-csv
#!/usr/bin/env python
import csv
import sys
from tqdm import tqdm

# allow large content in the dump
csv.field_size_limit(sys.maxsize)

class WikiDataParser:
    def __init__(self, args):
        self.args = args

    def _is_insert(self, line):
        """
        Returns true if the line begins a SQL insert statement.
        """
        return line.startswith('INSERT INTO') or False


    def _get_values(self, line):
        """
        Returns the portion of an INSERT statement containing values
        """
        return line.partition('` VALUES ')[2]


    def _values_sanity_check(self, values):
        """
        Ensures that values from the INSERT statement meet basic checks.
        """
        assert values
        assert values[0] == '('
        # Assertions have not been raised
        return True


    def _parse_values(self, values):
        """
        Given a file handle and the raw values from a MySQL INSERT
        statement, write the equivalent CSV to the file
        """
        results = []
        latest_row = []

        reader = csv.reader([values], delimiter=',',
                            doublequote=False,
                            escapechar='\\',
                            quotechar="'",
                            strict=True
        )
        
        for reader_row in reader:
            for column in reader_row:
                # If our current string is empty...
                if len(column) == 0 or column == 'NULL':
                    latest_row.append(chr(0))
                    continue
                # If our string starts with an open paren
                if column[0] == "(":
                    # Assume that this column does not begin
                    # a new row.
                    new_row = False
                    # If we've been filling out a row
                    if len(latest_row) > 0:
                        # Check if the previous entry ended in
                        # a close paren. If so, the row we've
                        # been filling out has been COMPLETED
                        # as:
                        #    1) the previous entry ended in a )
                        #    2) the current entry starts with a (
                        if latest_row[-1][-1] == ")":
                            # Remove the close paren.
                            latest_row[-1] = latest_row[-1][:-1]
                            new_row = True
                    # If we've found a new row, write it out
                    # and begin our new one
                    if new_row:
                        results.append(latest_row)
                        latest_row = []
                    # If we're beginning a new row, eliminate the
                    # opening parentheses.
                    if len(latest_row) == 0:
                        column = column[1:]
                # Add our column to the row we're working on.
                latest_row.append(column)
            # At the end of an INSERT statement, we'll
            # have the semicolon.
            # Make sure to remove the semicolon and
            # the close paren.
            if latest_row[-1][-2:] == ");":
                latest_row[-1] = latest_row[-1][:-2]
                results.append(latest_row)
        return results

    def construct_wikiid2wikidataid(self):
        with open(self.args.wikidata_dump_file, "r", errors='ignore') as f:
            lines = f.readlines()
        
        results = {}
        for line in tqdm(lines, desc="Processing WikiData..."):
            # Look for an INSERT statement and parse it.
            if self._is_insert(line):
                values = self._get_values(line)
                if self._values_sanity_check(values):
                    props = self._parse_values(values)
                    results.update({ \
                        p[0]: p[2] for p in props \
                            if p[1] == "wikibase_item" \
                    })
        return results
