const express = require("express");
const cors = require("cors"); // Import the cors middleware
const app = express();
const fs = require('fs/promises'); 
const bodyParser = require('body-parser');

app.use(cors()); // Enable CORS for all routes
app.use(express.json());

const filePath = './leaderboard.json';
app.use(bodyParser.json());

// Function to check if the provided password matches the stored password for a user
function isPasswordValid(player, providedPassword) {
  return player.password === providedPassword;
}


app.post('/updatescore', async (req, res) => {
  try {
    // Read the current leaderboard data from the JSON file
    const fileContent = await fs.readFile(filePath, 'utf8');
    const leaderboardData = JSON.parse(fileContent);

    // Extract player name, new score, and password from the request body
    const { playerName, newScore, password } = req.body;

    // Find the player in the leaderboard
    const playerIndex = leaderboardData.players.findIndex(
      (player) => player.name === playerName
    );

    if (playerIndex !== -1) {
      // If the player exists, check if the provided password is valid
      if (isPasswordValid(leaderboardData.players[playerIndex], password)) {
        // Update the score if the password is valid
        leaderboardData.players[playerIndex].score = newScore;
        // Save the updated leaderboard data back to the JSON file
        await fs.writeFile(filePath, JSON.stringify(leaderboardData, null, 2), 'utf8');

        res.status(200).json({ message: 'Player score updated successfully' });
      } else {
        res.status(403).json({ error: 'Invalid password' });
      }
    } else {
      // If the player doesn't exist, add a new player
      leaderboardData.players.push({
        name: playerName,
        score: newScore,
        password: password, // Store the password for the new player
      });

      // Save the updated leaderboard data back to the JSON file
      await fs.writeFile(filePath, JSON.stringify(leaderboardData, null, 2), 'utf8');

      res.status(200).json({ message: 'New player added successfully' });
    }
  } catch (error) {
    console.error('Error updating player score:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

  app.post('/deleteallusers', async (req, res) => {
    try {
      // Create an empty array for the leaderboard
      const emptyLeaderboard = {
        players: [],
      };
  
      // Write the empty leaderboard to the JSON file
      await fs.writeFile(filePath, JSON.stringify(emptyLeaderboard, null, 2), 'utf8');
  
      res.status(200).json({ message: 'All users deleted successfully' });
    } catch (error) {
      console.error('Error deleting all users:', error);
      res.status(500).json({ error: 'Internal Server Error' });
    }
  });
 
  app.get('/leaderboard', async (req, res) => {
    try {
      // Read the file using fs/promises
      const data = await fs.readFile('leaderboard.json', 'utf8');
  
      // Parse the JSON data
      const jsonData = JSON.parse(data);
  
      // Send the JSON data as a response
      res.status(200).json(jsonData);
    } catch (error) {
      console.error('Error reading or parsing JSON:', error);
      res.status(500).send('Internal Server Error');
    }
  });


// Track progress

const counterFilePath = './nameCounter.json';

// Function to read the name counter from the JSON file
async function readNameCounter() {
  try {
    const fileContent = await fs.readFile(counterFilePath, 'utf8');
    return JSON.parse(fileContent);
  } catch (error) {
    return {};
  }
}

// Function to write the name counter to the JSON file
async function writeNameCounter(nameCounter) {
  await fs.writeFile(counterFilePath, JSON.stringify(nameCounter, null, 2), 'utf8');
}

app.post('/countname', async (req, res) => {
  try {
    const { name } = req.body;

    if (!name) {
      return res.status(400).json({ error: 'Name parameter is required' });
    }

    // Read the current name counter from the JSON file
    const nameCounter = await readNameCounter();

    // Increment the counter for the specified name
    nameCounter[name] = (nameCounter[name] || 0) + 1;

    // Write the updated name counter back to the JSON file
    await writeNameCounter(nameCounter);

    res.status(200).json({ message: `Name ${name} has been called ${nameCounter[name]} times` });
  } catch (error) {
    console.error('Error counting name:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/namecounts', async (req, res) => {
  try {
    // Read the current name counter from the JSON file
    const nameCounter = await readNameCounter();

    res.status(200).json(nameCounter);
  } catch (error) {
    console.error('Error retrieving name counts:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// New POST endpoint to empty all counts
app.post('/emptycounts', async (req, res) => {
  try {
    // Write an empty object to the JSON file
    await writeNameCounter({});

    res.status(200).json({ message: 'All name counts have been emptied' });
  } catch (error) {
    console.error('Error emptying name counts:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});


 
const port = process.env.PORT || 443;
 
app.listen(port, () => console.log(`Server hört an Port: ${port}`));