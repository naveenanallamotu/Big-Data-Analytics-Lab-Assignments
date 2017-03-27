import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import static org.openimaj.image.DisplayUtilities.displayName;

/**
 * Created by JyothiKiran on 3/24/2017.
 */

@WebServlet(name = "clarifai", urlPatterns = "/clarifai")
public class clarifai extends HttpServlet {

    @Override
    public void doPost(HttpServletRequest req, HttpServletResponse resp) throws IOException {

        resp.setHeader("Access-Control-Allow-Origin", "*");
        resp.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE");
        resp.setHeader("Access-Control-Max-Age", "3600");
        resp.setHeader("Access-Control-Allow-Headers", "x-requested-with, X-Auth-Token, Content-Type");


        resp.setContentType("text/event-stream");

        //encoding must be set to UTF-8
        resp.setCharacterEncoding("UTF-8");

        PrintWriter writer = resp.getWriter();

        for (int i = 0; i < 10; i++) {

            writer.write("data: " + System.currentTimeMillis() + "\n\n");

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        writer.close();

    }

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {

        resp.setHeader("Access-Control-Allow-Origin", "*");
        resp.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE");
        resp.setHeader("Access-Control-Max-Age", "3600");
        resp.setHeader("Access-Control-Allow-Headers", "x-requested-with, X-Auth-Token, Content-Type");

        resp.setContentType("text/event-stream");

        //encoding must be set to UTF-8
        resp.setCharacterEncoding("UTF-8");

        final ClarifaiClient client = new ClarifaiBuilder("KKQIegBW9uOl_3vaMSzqq4QCfPNyNBvB7XNBz1vE", "xsY48eiDhhsFo5M7HE3F71ZYkB_tEQmemlWekTgG")
                .client(new OkHttpClient()) // OPTIONAL. Allows customization of OkHttp by the user
                .buildSync(); // or use .build() to get a Future<ClarifaiClient>
        client.getToken();


        Data data_ob = Data.getInstance();
        PrintWriter writer = resp.getWriter();

        File file = new File("Input/Crossing");
        File[] files = file.listFiles();
        for (int i = 0; i < files.length; i++) {
            ClarifaiResponse response = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(files[i]))
                    )
                    .executeSync();
            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response.get();
            MBFImage image = ImageUtilities.readMBF(files[i]);
            int x = image.getWidth();
            int y = image.getHeight();

            System.out.println("*************" + files[i] + "***********");
            List<Concept> data = predictions.get(0).data();
            for (int j = 0; j < data.size(); j++) {
                System.out.println(data.get(j).name() + " - " + data.get(j).value());
                image.drawText(data.get(j).name(), (int) Math.floor(Math.random() * x), (int) Math.floor(Math.random() * y), HersheyFont.ASTROLOGY, 20, RGBColour.RED);
            }
            JFrame abc = displayName(image, "image" + i);
            DisplayUtilities.display(image, abc).toString();


            for (long k = 0; k < 100000000; i++) {
                String data1 = data_ob.getData1(DisplayUtilities.display(image, abc).toString());
//            if (data_ob.getFlag()){
                writer.write("data: " + data1 + "\n\n");
                data_ob.setFlag(false);
                System.out.println("*********Set FLAG**************");
//            }

                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                writer.flush();
            }
            writer.close();
        }
    }
}