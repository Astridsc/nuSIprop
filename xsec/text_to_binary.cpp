#include <fstream>
#include <cstring>

// Convert text files to binary files

int main(){
  // Process alpha_phiphi.dat
  {
    FILE *fp = fopen("alpha_phiphi.dat", "r");
    FILE *f = fopen("alpha_phiphi.bin", "wb");

    char line[1000]; // Line to be scanned
    char *var; // Each variable to be read
    constexpr int n_lines = 100000000; // Total number of lines to be read [excluding comments]
    constexpr int n_els = 4;
    constexpr int chunk_size = 1000000; // Process 1 million lines at a time
    
    float** data = new float*[chunk_size];
    for(int i=0; i<chunk_size; ++i)
      data[i] = new float[n_els];

    /* We read the data file in chunks */
    int total_lines_processed = 0;
    while (total_lines_processed < n_lines) {
      int lines_in_chunk = 0;
      
      // Read a chunk of lines
      while (lines_in_chunk < chunk_size && total_lines_processed + lines_in_chunk < n_lines) {
        fgets(line, sizeof line, fp);
        if (line[0] == '#'){ // Ignore lines starting with #
          continue; 
        }
        
        var = strtok(line, " "); // Read first element in line
        for(int i=0; i<n_els; ++i){
          sscanf(var, "%f", &data[lines_in_chunk][i]);
          var = strtok(NULL, " "); // Read next element in line
        }
        lines_in_chunk++;
      }

      // Write the chunk to binary file
      for(int j=0; j<lines_in_chunk; ++j)
        fwrite(data[j], sizeof(float), n_els, f);
      
      total_lines_processed += lines_in_chunk;
    }
    fclose(fp);
    fclose(f);

    for(int i=0; i<chunk_size; ++i)
      delete[] data[i];
    delete[] data;
  }

  // Process alphatilde_phiphi.dat
  {
    FILE *fp = fopen("alphatilde_phiphi.dat", "r");
    FILE *f = fopen("alphatilde_phiphi.bin", "wb");

    char line[1000]; // Line to be scanned
    char *var; // Each variable to be read
    constexpr int n_lines_tilde = 500000; // Total number of lines to be read [excluding comments]
    constexpr int n_els_tilde = 3;
    constexpr int chunk_size = 100000; // Process 100k lines at a time
    
    float** data = new float*[chunk_size];
    for(int i=0; i<chunk_size; ++i)
      data[i] = new float[n_els_tilde];

    /* We read the data file in chunks */
    int total_lines_processed = 0;
    while (total_lines_processed < n_lines_tilde) {
      int lines_in_chunk = 0;
      
      // Read a chunk of lines
      while (lines_in_chunk < chunk_size && total_lines_processed + lines_in_chunk < n_lines_tilde) {
        fgets(line, sizeof line, fp);
        if (line[0] == '#'){ // Ignore lines starting with #
          continue; 
        }
        
        var = strtok(line, " "); // Read first element in line
        for(int i=0; i<n_els_tilde; ++i){
          sscanf(var, "%f", &data[lines_in_chunk][i]);
          var = strtok(NULL, " "); // Read next element in line
        }
        lines_in_chunk++;
      }

      // Write the chunk to binary file
      for(int j=0; j<lines_in_chunk; ++j)
        fwrite(data[j], sizeof(float), n_els_tilde, f);
      
      total_lines_processed += lines_in_chunk;
    }
    fclose(fp);
    fclose(f);

    for(int i=0; i<chunk_size; ++i)
      delete[] data[i];
    delete[] data;
  }

  return 0;
}
